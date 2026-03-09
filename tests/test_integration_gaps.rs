//! Wave U — closing the two remaining SymPy cross-validation integration gaps.
//!
//! U1: Negative trig power integration (sec², csc², sec⁴, …)
//! U2: Cyclic integration by parts (exp·sin, exp·cos)
//!
//! Every test verifies the Fundamental Theorem of Calculus numerically:
//!   d/dx(∫ f dx) ≈ f   at a non-trivial test point.

use symplex::prelude::*;

/// Verify FTC: the derivative of the antiderivative must match the integrand
/// at a test point.  Also asserts the antiderivative is *not* unevaluated.
fn assert_ftc(integrand: &Ex, var: &Ex, label: &str) {
    let ctx = integrand.context();
    let anti = integrand.integrate(var);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "{label}: unevaluated: {s}");

    let deriv = anti.diff(var);
    let test_point = ctx.rational(7, 10);
    if let (Ok(o), Ok(d)) = (
        integrand.subs(var, &test_point).eval_f64(),
        deriv.subs(var, &test_point).eval_f64(),
    )
        && o.is_finite() && d.is_finite()
    {
        let diff = (o - d).abs();
        let tol = 1e-8 * o.abs().max(1.0);
        assert!(
            diff < tol,
            "{label}: FTC failed — integrand={o:.12}, deriv={d:.12}, diff={diff:.2e}",
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// U1 — Negative trig powers
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sec_squared() {
    let ctx = Context::new();
    // ∫ sec²(x) dx = ∫ cos(x)^(-2) dx = tan(x)
    let x = ctx.symbol("x");
    assert_ftc(&x.cos().powi(-2), &x, "∫sec²(x)dx");
}

#[test]
fn integrate_csc_squared() {
    let ctx = Context::new();
    // ∫ csc²(x) dx = ∫ sin(x)^(-2) dx = −cot(x)
    let x = ctx.symbol("x");
    assert_ftc(&x.sin().powi(-2), &x, "∫csc²(x)dx");
}

#[test]
fn integrate_sec_x() {
    let ctx = Context::new();
    // ∫ sec(x) dx = ∫ cos(x)^(-1) dx = ln|sec(x)+tan(x)|
    let x = ctx.symbol("x");
    assert_ftc(&x.cos().powi(-1), &x, "∫sec(x)dx");
}

#[test]
fn integrate_csc_x() {
    let ctx = Context::new();
    // ∫ csc(x) dx = ∫ sin(x)^(-1) dx = −ln|csc(x)+cot(x)|
    let x = ctx.symbol("x");
    assert_ftc(&x.sin().powi(-1), &x, "∫csc(x)dx");
}

#[test]
fn integrate_sec_fourth() {
    let ctx = Context::new();
    // ∫ sec⁴(x) dx = ∫ cos(x)^(-4) dx — two reduction steps
    let x = ctx.symbol("x");
    assert_ftc(&x.cos().powi(-4), &x, "∫sec⁴(x)dx");
}

#[test]
fn integrate_csc_fourth() {
    let ctx = Context::new();
    // ∫ csc⁴(x) dx = ∫ sin(x)^(-4) dx — two reduction steps
    let x = ctx.symbol("x");
    assert_ftc(&x.sin().powi(-4), &x, "∫csc⁴(x)dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// U2 — Cyclic integration by parts (exp · trig)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_exp_sin() {
    let ctx = Context::new();
    // ∫ exp(x)·sin(x) dx = exp(x)(sin(x)−cos(x))/2
    let x = ctx.symbol("x");
    let integrand = &x.exp() * &x.sin();
    assert_ftc(&integrand, &x, "∫exp(x)sin(x)dx");
}

#[test]
fn integrate_exp_cos() {
    let ctx = Context::new();
    // ∫ exp(x)·cos(x) dx = exp(x)(sin(x)+cos(x))/2
    let x = ctx.symbol("x");
    let integrand = &x.exp() * &x.cos();
    assert_ftc(&integrand, &x, "∫exp(x)cos(x)dx");
}
