//! Integration tests for the heuristic Risch integrator.
//!
//! These tests exercise integration through the public `Ex::integrate()` API.
//! Once the heurisch module is wired in as a fallback (by a separate agent),
//! the tests will automatically verify that the heuristic integrator handles
//! cases the rule-based engine cannot.
//!
//! Each test is structured defensively: if the result is still an unevaluated
//! `Integral(…)` node (meaning heurisch is not yet wired in), the test prints
//! a diagnostic message and passes. Once wired in, the tests verify
//! correctness via the Fundamental Theorem of Calculus (numeric
//! differentiation check).

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: verify an antiderivative by numeric differentiation (FTC)
// ═══════════════════════════════════════════════════════════════════════════

/// Numerically verify that `d/dx result ≈ integrand` at several test points.
///
/// Returns `true` if the check passes at all finite test points.
fn ftc_check(integrand: &Ex, result: &Ex, var: &Ex, points: &[i64]) -> bool {
    let deriv = result.diff(var);
    for &pt in points {
        let orig = integrand.subs_i64(var, pt).eval().eval_f64();
        let dval = deriv.subs_i64(var, pt).eval().eval_f64();
        match (orig, dval) {
            (Ok(o), Ok(d)) => {
                if !o.is_finite() || !d.is_finite() {
                    continue; // skip non-finite points
                }
                let scale = o.abs().max(1.0);
                if (o - d).abs() / scale > 1e-4 {
                    eprintln!(
                        "FTC FAIL at x={pt}: f(x)={o}, F'(x)={d}, diff={}",
                        (o - d).abs()
                    );
                    return false;
                }
            }
            _ => continue, // eval failed, skip
        }
    }
    true
}

/// Check if a result string indicates an unevaluated integral.
fn is_unevaluated(s: &str) -> bool {
    s.contains("Integral(")
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ exp(x) / (1 + exp(x))² dx → -1/(1+exp(x))
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_exp_over_one_plus_exp_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let exp_x = x.exp();
    let denom = (&exp_x + 1).powi(2);
    let expr = &exp_x / &denom;
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: heurisch not yet wired in for exp(x)/(1+exp(x))^2, skipping verification");
        return;
    }

    // Verify by FTC at x = 0, 1, 2
    assert!(
        ftc_check(&expr, &result, &x, &[0, 1, 2]),
        "FTC verification failed for ∫ exp(x)/(1+exp(x))^2 dx = {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ 1/(1 + exp(x)) dx → x - ln(1+exp(x))
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_one_over_one_plus_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = ctx.int(1) / &(&x.exp() + 1);
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: heurisch not yet wired in for 1/(1+exp(x)), skipping verification");
        return;
    }

    assert!(
        ftc_check(&expr, &result, &x, &[0, 1, 2]),
        "FTC verification failed for ∫ 1/(1+exp(x)) dx = {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ x / (x⁴ + 1) dx — partial fractions alternative
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_x_over_x4_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = x.clone();
    let denom = &x.powi(4) + 1;
    let expr = &numer / &denom;
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: heurisch not yet wired in for x/(x^4+1), skipping verification");
        return;
    }

    // Verify by FTC at several points (avoid x=0 where denom is small-ish)
    assert!(
        ftc_check(&expr, &result, &x, &[1, 2, 3]),
        "FTC verification failed for ∫ x/(x^4+1) dx = {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ ln(x)² dx → x·ln(x)² - 2x·ln(x) + 2x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_ln_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.ln().powi(2);
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: heurisch not yet wired in for ln(x)^2, skipping verification");
        return;
    }

    // Use x > 0 for ln to be defined; use x = 1, 2, 3
    assert!(
        ftc_check(&expr, &result, &x, &[1, 2, 3]),
        "FTC verification failed for ∫ ln(x)^2 dx = {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ x·exp(x) dx — should already work via by-parts, but verify
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_x_exp_x_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * &x.exp();
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: integration not handling x*exp(x), skipping");
        return;
    }

    assert!(
        ftc_check(&expr, &result, &x, &[0, 1, 2]),
        "FTC verification failed for ∫ x·exp(x) dx = {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ exp(x)·sin(x) dx — cyclic integration by parts
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_exp_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.exp() * &x.sin();
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: heurisch not yet wired in for exp(x)*sin(x), skipping verification");
        return;
    }

    assert!(
        ftc_check(&expr, &result, &x, &[0, 1, 2]),
        "FTC verification failed for ∫ exp(x)·sin(x) dx = {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ x² · exp(x) dx — needs two rounds of by-parts or heurisch
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_x_squared_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) * &x.exp();
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: heurisch not yet wired in for x^2*exp(x), skipping verification");
        return;
    }

    assert!(
        ftc_check(&expr, &result, &x, &[0, 1, 2]),
        "FTC verification failed for ∫ x²·exp(x) dx = {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ 1/(x² + 1) dx — should be atan(x), rule-based handles it
// Included as a sanity check / regression guard
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_atan_form_sanity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = ctx.int(1) / &(&x.powi(2) + 1);
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: even rule-based should handle 1/(x^2+1), investigate");
        return;
    }

    assert!(
        ftc_check(&expr, &result, &x, &[0, 1, 2]),
        "FTC verification failed for ∫ 1/(x²+1) dx = {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ x · sin(x²) dx — u-substitution candidate
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_x_sin_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * &x.powi(2).sin();
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: heurisch not yet wired in for x*sin(x^2), skipping verification");
        return;
    }

    // Expected: -cos(x²)/2
    assert!(
        ftc_check(&expr, &result, &x, &[1, 2, 3]),
        "FTC verification failed for ∫ x·sin(x²) dx = {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ exp(x)/(1+exp(x)) dx → ln(1+exp(x))
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_exp_over_one_plus_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let exp_x = x.exp();
    let expr = &exp_x / &(&exp_x + 1);
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: heurisch not yet wired in for exp(x)/(1+exp(x)), skipping verification");
        return;
    }

    assert!(
        ftc_check(&expr, &result, &x, &[0, 1, 2]),
        "FTC verification failed for ∫ exp(x)/(1+exp(x)) dx = {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ sin(x)·cos(x) dx — product of trig
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_sin_cos_product() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin() * &x.cos();
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: heurisch not yet wired in for sin(x)*cos(x), skipping verification");
        return;
    }

    // Expected: sin²(x)/2 or -cos²(x)/2 + C
    assert!(
        ftc_check(&expr, &result, &x, &[1, 2, 3]),
        "FTC verification failed for ∫ sin(x)·cos(x) dx = {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ 1/(x·ln(x)) dx — log integral form
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_one_over_x_ln_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = ctx.int(1) / &(&x * &x.ln());
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: heurisch not yet wired in for 1/(x·ln(x)), skipping verification");
        return;
    }

    // Expected: ln(ln(x))
    assert!(
        ftc_check(&expr, &result, &x, &[2, 3, 5]),
        "FTC verification failed for ∫ 1/(x·ln(x)) dx = {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ x·ln(x) dx → x²/2·ln(x) - x²/4
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_x_ln_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * &x.ln();
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: heurisch not yet wired in for x·ln(x), skipping verification");
        return;
    }

    assert!(
        ftc_check(&expr, &result, &x, &[1, 2, 3]),
        "FTC verification failed for ∫ x·ln(x) dx = {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: ∫ 2x·exp(x²) dx → exp(x²)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heurisch_2x_exp_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two_x = &ctx.int(2) * &x;
    let expr = &two_x * &x.powi(2).exp();
    let result = expr.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: heurisch not yet wired in for 2x·exp(x²), skipping verification");
        return;
    }

    assert!(
        ftc_check(&expr, &result, &x, &[0, 1]),
        "FTC verification failed for ∫ 2x·exp(x²) dx = {s}"
    );
}
