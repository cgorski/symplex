//! Tests for piecewise parametric wrapping and special function integration.
//!
//! These tests verify that:
//! 1. Integrals with symbolic parameters produce `Piecewise` results with
//!    explicit `Ne` conditions for degenerate parameter values.
//! 2. Integrals matching special function patterns (Si, Ci, Ei, li) produce
//!    the corresponding `Apply` nodes instead of unevaluated `Integral`.
//! 3. Purely numeric integrands do NOT produce spurious `Piecewise` wrappers.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn check_contains(expr: &Ex, needle: &str, label: &str) {
    let s = format!("{expr}");
    assert!(
        s.contains(needle),
        "{label}: expected output to contain `{needle}`, got: {s}"
    );
}

fn check_not_contains(expr: &Ex, needle: &str, label: &str) {
    let s = format!("{expr}");
    assert!(
        !s.contains(needle),
        "{label}: expected output NOT to contain `{needle}`, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Feature 1 — Piecewise parametric wrapping
// ═══════════════════════════════════════════════════════════════════════════

/// ∫ sin(a·x) dx should be Piecewise with Ne(a, 0).
///
/// Generic result: -cos(a·x)/a
/// Degenerate at a=0: integrand becomes sin(0)=0, integral is 0.
/// Expected: Piecewise((-cos(a*x)/a, a != 0), (0, True))
#[test]
fn piecewise_sin_ax() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let ax = &a * &x;
    let integrand = ax.sin();
    let result = integrand.integrate(&x);
    let _s = format!("{result}");

    // Should be a Piecewise expression
    check_contains(&result, "Piecewise", "∫sin(a*x)dx");
    // Should contain the Ne(a, 0) condition
    check_contains(&result, "!=", "∫sin(a*x)dx condition");
    // Should contain cos (from the generic branch)
    check_contains(&result, "cos", "∫sin(a*x)dx generic branch");
    // Should NOT be an unevaluated Integral
    check_not_contains(&result, "Integral", "∫sin(a*x)dx");

    // Verify numeric evaluation at a=2, x=1 matches the generic branch
    let val = result
        .subs(&a, &ctx.int(2))
        .subs(&x, &ctx.int(1));
    if let Ok(v) = val.eval_f64() {
        let expected = -(2.0_f64).cos() / 2.0;
        assert!(
            (v - expected).abs() < 1e-10,
            "∫sin(2x)dx at x=1: got {v}, expected {expected}"
        );
    }
}

/// ∫ exp(a·x) dx should be Piecewise with Ne(a, 0).
///
/// Generic result: exp(a·x)/a
/// Degenerate at a=0: integrand becomes exp(0)=1, integral is x.
/// Expected: Piecewise((exp(a*x)/a, a != 0), (x, True))
#[test]
fn piecewise_exp_ax() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let ax = &a * &x;
    let integrand = ax.exp();
    let result = integrand.integrate(&x);
    let _s = format!("{result}");

    check_contains(&result, "Piecewise", "∫exp(a*x)dx");
    check_contains(&result, "!=", "∫exp(a*x)dx condition");
    check_contains(&result, "exp", "∫exp(a*x)dx generic branch");
    check_not_contains(&result, "Integral", "∫exp(a*x)dx");

    // Verify numeric evaluation at a=3, x=1 matches the generic branch
    let val = result
        .subs(&a, &ctx.int(3))
        .subs(&x, &ctx.int(1));
    if let Ok(v) = val.eval_f64() {
        let expected = (3.0_f64).exp() / 3.0;
        assert!(
            (v - expected).abs() < 1e-6,
            "∫exp(3x)dx at x=1: got {v}, expected {expected}"
        );
    }
}

/// ∫ x^n dx should be Piecewise with Ne(n, -1).
///
/// Generic result: x^(n+1)/(n+1)
/// Degenerate at n=-1: integrand is 1/x, integral is ln|x|.
/// Expected: Piecewise((x^(n+1)/(n+1), n != -1), (ln|x|, True))
#[test]
fn piecewise_x_to_n() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let n = ctx.symbol("n");
    let integrand = x.pow(&n);
    let result = integrand.integrate(&x);
    let _s = format!("{result}");

    // Should be Piecewise (n+1 is in denominator, degenerate at n=-1)
    check_contains(&result, "Piecewise", "∫x^n dx");
    check_contains(&result, "!=", "∫x^n dx condition");
    check_not_contains(&result, "Integral", "∫x^n dx");

    // Verify numeric evaluation at n=2, x=3: x^3/3 = 9
    let val = result
        .subs(&n, &ctx.int(2))
        .subs(&x, &ctx.int(3));
    if let Ok(v) = val.eval_f64() {
        let expected = 27.0 / 3.0; // 3^3 / 3 = 9
        assert!(
            (v - expected).abs() < 1e-10,
            "∫x^2 dx at x=3: got {v}, expected {expected}"
        );
    }
}

/// ∫ sin(2·x) dx should NOT have Piecewise (no free params).
///
/// The coefficient 2 is numeric, so the result should be a plain
/// expression -cos(2x)/2 with no Piecewise wrapper.
#[test]
fn no_piecewise_for_numeric() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    let integrand = (&two * &x).sin();
    let result = integrand.integrate(&x);

    check_not_contains(&result, "Piecewise", "∫sin(2x)dx");
    check_not_contains(&result, "Integral", "∫sin(2x)dx");
    check_contains(&result, "cos", "∫sin(2x)dx");

    // Verify numerically: -cos(2)/2
    let val = result.subs(&x, &ctx.int(1));
    if let Ok(v) = val.eval_f64() {
        let expected = -(2.0_f64).cos() / 2.0;
        assert!(
            (v - expected).abs() < 1e-10,
            "∫sin(2x)dx at x=1: got {v}, expected {expected}"
        );
    }
}

/// ∫ 1/(a·x+b) dx should be Piecewise with Ne(a, 0).
///
/// Generic result: ln|a*x+b|/a
/// Degenerate at a=0: integrand becomes 1/b, integral is x/b.
/// Expected: Piecewise((ln|a*x+b|/a, a != 0), (x/b, True))
#[test]
fn piecewise_1_over_ax_plus_b() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let inner = &(&a * &x) + &b;
    let integrand = inner.powi(-1);
    let result = integrand.integrate(&x);
    let _s = format!("{result}");

    // Should not be an unevaluated Integral
    check_not_contains(&result, "Integral", "∫1/(a*x+b) dx");
    // Should contain ln (from the generic branch)
    check_contains(&result, "ln", "∫1/(a*x+b) dx generic branch");
    // Should be Piecewise with a != 0
    check_contains(&result, "Piecewise", "∫1/(a*x+b) dx");
    check_contains(&result, "!=", "∫1/(a*x+b) dx condition");

    // Verify numeric evaluation at a=2, b=1, x=1: ln|2+1|/2 = ln(3)/2
    let val = result
        .subs(&a, &ctx.int(2))
        .subs(&b, &ctx.int(1))
        .subs(&x, &ctx.int(1));
    if let Ok(v) = val.eval_f64() {
        let expected = (3.0_f64).ln() / 2.0;
        assert!(
            (v - expected).abs() < 1e-10,
            "∫1/(2x+1)dx at x=1: got {v}, expected {expected}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Feature 2 — Special function integration table
// ═══════════════════════════════════════════════════════════════════════════

/// ∫ sin(x)/x dx should produce Si(x), not an unevaluated Integral.
#[test]
fn special_func_si() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.sin() / &x;
    let result = integrand.integrate(&x);
    let _s = format!("{result}");

    check_not_contains(&result, "Integral", "∫sin(x)/x dx");
    check_contains(&result, "Si", "∫sin(x)/x dx");
}

/// ∫ exp(x)/x dx should produce Ei(x), not an unevaluated Integral.
#[test]
fn special_func_ei() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.exp() / &x;
    let result = integrand.integrate(&x);
    let _s = format!("{result}");

    check_not_contains(&result, "Integral", "∫exp(x)/x dx");
    check_contains(&result, "Ei", "∫exp(x)/x dx");
}

/// ∫ 1/ln(x) dx should produce li(x), not an unevaluated Integral.
#[test]
fn special_func_li() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let integrand = &one / &x.ln();
    let result = integrand.integrate(&x);
    let _s = format!("{result}");

    check_not_contains(&result, "Integral", "∫1/ln(x) dx");
    check_contains(&result, "li", "∫1/ln(x) dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional edge cases and regression tests
// ═══════════════════════════════════════════════════════════════════════════

/// ∫ cos(x)/x dx should produce Ci(x).
#[test]
fn special_func_ci() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.cos() / &x;
    let result = integrand.integrate(&x);
    let _s = format!("{result}");

    check_not_contains(&result, "Integral", "∫cos(x)/x dx");
    check_contains(&result, "Ci", "∫cos(x)/x dx");
}

/// Piecewise wrapping should not interfere with purely numeric integrals.
/// ∫ exp(3x) dx = exp(3x)/3 with no Piecewise.
#[test]
fn no_piecewise_exp_numeric_coeff() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let three = ctx.int(3);
    let integrand = (&three * &x).exp();
    let result = integrand.integrate(&x);

    check_not_contains(&result, "Piecewise", "∫exp(3x)dx");
    check_not_contains(&result, "Integral", "∫exp(3x)dx");
    check_contains(&result, "exp", "∫exp(3x)dx");
}

/// Piecewise wrapping should not interfere with ∫ cos(x) dx = sin(x).
/// No parameters, no Piecewise.
#[test]
fn no_piecewise_cos_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.cos();
    let result = integrand.integrate(&x);

    check_not_contains(&result, "Piecewise", "∫cos(x)dx");
    check_not_contains(&result, "Integral", "∫cos(x)dx");
    check_contains(&result, "sin", "∫cos(x)dx");
}

/// ∫ cos(a·x) dx should be Piecewise with Ne(a, 0).
#[test]
fn piecewise_cos_ax() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let ax = &a * &x;
    let integrand = ax.cos();
    let result = integrand.integrate(&x);

    check_contains(&result, "Piecewise", "∫cos(a*x)dx");
    check_contains(&result, "!=", "∫cos(a*x)dx condition");
    check_contains(&result, "sin", "∫cos(a*x)dx generic branch");
    check_not_contains(&result, "Integral", "∫cos(a*x)dx");
}

/// FTC sanity check: d/dx(∫ sin(a·x) dx) ≈ sin(a·x) at a=2, x=1.
/// This verifies the generic branch of the Piecewise is mathematically correct.
#[test]
fn piecewise_sin_ax_ftc() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let ax = &a * &x;
    let integrand = ax.sin();
    let anti = integrand.integrate(&x);
    let deriv = anti.diff(&x);

    // Substitute a=2, x=1 and compare
    let a_val = ctx.int(2);
    let x_val = ctx.int(1);

    let orig = integrand
        .subs(&a, &a_val)
        .subs(&x, &x_val);
    let diff = deriv
        .subs(&a, &a_val)
        .subs(&x, &x_val);

    if let (Ok(o), Ok(d)) = (orig.eval_f64(), diff.eval_f64()) {
        let err = (o - d).abs();
        assert!(
            err < 1e-8,
            "FTC for ∫sin(a*x)dx at a=2,x=1: integrand={o}, deriv={d}, err={err}"
        );
    }
}

/// FTC sanity check for exp(a·x): d/dx(∫ exp(a·x) dx) ≈ exp(a·x) at a=3, x=1.
#[test]
fn piecewise_exp_ax_ftc() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let ax = &a * &x;
    let integrand = ax.exp();
    let anti = integrand.integrate(&x);
    let deriv = anti.diff(&x);

    let a_val = ctx.int(3);
    let x_val = ctx.int(1);

    let orig = integrand
        .subs(&a, &a_val)
        .subs(&x, &x_val);
    let diff = deriv
        .subs(&a, &a_val)
        .subs(&x, &x_val);

    if let (Ok(o), Ok(d)) = (orig.eval_f64(), diff.eval_f64()) {
        let err = (o - d).abs();
        let tol = 1e-8 * o.abs().max(1.0);
        assert!(
            err < tol,
            "FTC for ∫exp(a*x)dx at a=3,x=1: integrand={o}, deriv={d}, err={err}"
        );
    }
}
