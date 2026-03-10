//! End-to-end Risch integration tests.
//!
//! These tests exercise the full `expr.integrate(&x)` path — from the
//! public `Ex` API through the heuristic integrator and the Risch
//! rational-function pipeline (Hermite + Rothstein-Trager), and verify
//! results by numerical FTC checks:
//!
//!   d/dx(∫ f dx) ≈ f   at multiple test points
//!
//! This catches mathematical errors that structural tests miss.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Verify the Fundamental Theorem of Calculus numerically:
///   d/dx(result) ≈ integrand  at several test points.
///
/// `points` are integer x-values to test at.  We substitute each,
/// evaluate both sides to f64, and compare within `tol`.
fn verify_ftc(
    integrand: &Ex,
    result: &Ex,
    x: &Ex,
    points: &[i64],
    tol: f64,
    label: &str,
) {
    let d_result = result.diff(x);
    let mut checked = 0;
    for &pt in points {
        let lhs = d_result.eval_f64_with(&[(x, pt)]);
        let rhs = integrand.eval_f64_with(&[(x, pt)]);
        match (lhs, rhs) {
            (Ok(l), Ok(r)) => {
                if l.is_finite() && r.is_finite() {
                    let err = (l - r).abs();
                    let scale = r.abs().max(1.0);
                    assert!(
                        err / scale < tol,
                        "{label}: FTC failed at x={pt}: d/dx(result)={l:.8}, integrand={r:.8}, err={err:.2e}"
                    );
                    checked += 1;
                }
            }
            _ => {
                // Evaluation failed at this point (singularity, etc.) — skip.
            }
        }
    }
    assert!(
        checked >= 1,
        "{label}: FTC check was vacuous — no point evaluated successfully"
    );
}

/// Check that an integral result does NOT contain unevaluated Integral nodes.
fn assert_evaluated(result: &Ex, label: &str) {
    assert!(
        !result.has_unevaluated(),
        "{label}: result contains unevaluated Integral node: {}",
        result
    );
}

// Test points that avoid 0 (singularity for 1/x) and ±1 (common roots).
const SAFE_POINTS: &[i64] = &[-3, -2, 2, 3, 5, 7];
// Test points that include 0 but avoid ±1.
const POINTS_WITH_ZERO: &[i64] = &[-3, -2, 0, 2, 3, 5];
// Strictly positive points (for ln(x)).
const POSITIVE_POINTS: &[i64] = &[2, 3, 5, 7, 10];

// ═══════════════════════════════════════════════════════════════════════════
// Group 4: End-to-end integration tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn e2e_polynomial() {
    // ∫ (x³ + 2x) dx = x⁴/4 + x²
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = expr!(ctx, x ^ 3 + 2 * x);
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ x³+2x dx");
    verify_ftc(&integrand, &result, &x, POINTS_WITH_ZERO, 1e-8, "∫ x³+2x dx");
}

#[test]
fn e2e_one_over_x() {
    // ∫ 1/x dx = ln|x|
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = &ctx.int(1) / &x;
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ 1/x dx");
    verify_ftc(&integrand, &result, &x, SAFE_POINTS, 1e-8, "∫ 1/x dx");
}

#[test]
fn e2e_one_over_x_squared() {
    // ∫ 1/x² dx = -1/x
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = &ctx.int(1) / &x.powi(2);
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ 1/x² dx");
    verify_ftc(&integrand, &result, &x, SAFE_POINTS, 1e-8, "∫ 1/x² dx");
}

#[test]
fn e2e_one_over_x_cubed() {
    // ∫ 1/x³ dx = -1/(2x²)   (Hermite reduction)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = &ctx.int(1) / &x.powi(3);
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ 1/x³ dx");
    verify_ftc(&integrand, &result, &x, SAFE_POINTS, 1e-8, "∫ 1/x³ dx");
}

#[test]
fn e2e_one_over_x_squared_minus_1() {
    // ∫ 1/(x²-1) dx = ½ ln|x-1| - ½ ln|x+1|  (partial fractions / Rothstein-Trager)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = &ctx.int(1) / &expr!(ctx, x ^ 2 - 1);
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ 1/(x²-1) dx");
    // Avoid x = ±1 (poles).
    verify_ftc(&integrand, &result, &x, &[-3, -2, 2, 3, 5], 1e-8, "∫ 1/(x²-1) dx");
}

#[test]
fn e2e_repeated_quadratic() {
    // ∫ 1/(x²+1)² dx = x/(2(x²+1)) + (1/2)·arctan(x)
    //
    // Hermite reduction extracts the rational part x/(2(x²+1)).
    // The remainder 1/(2(x²+1)) has algebraic residues (±i/2),
    // so Rothstein-Trager flags it as algebraic.  The recursive
    // integration then resolves it via the standard-form detector
    // (arctan).
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let denom = expr!(ctx, (x ^ 2 + 1) ^ 2);
    let integrand = &ctx.int(1) / &denom;
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ 1/(x²+1)² dx");
    verify_ftc(&integrand, &result, &x, POINTS_WITH_ZERO, 1e-8, "∫ 1/(x²+1)² dx");
}

#[test]
fn e2e_2x_plus_1_over_x_plus_1_squared() {
    // ∫ (2x+1)/(x+1)² dx — Hermite reduction + possible log part.
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let numer = expr!(ctx, 2 * x + 1);
    let denom = expr!(ctx, (x + 1) ^ 2);
    let integrand = &numer / &denom;
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ (2x+1)/(x+1)² dx");
    // Avoid x = -1 (pole).
    verify_ftc(&integrand, &result, &x, &[-3, -2, 0, 2, 3, 5], 1e-8, "∫ (2x+1)/(x+1)² dx");
}

#[test]
fn e2e_sin_x() {
    // ∫ sin(x) dx = -cos(x)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = x.sin();
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ sin(x) dx");
    verify_ftc(&integrand, &result, &x, POINTS_WITH_ZERO, 1e-8, "∫ sin(x) dx");
}

#[test]
fn e2e_cos_x() {
    // ∫ cos(x) dx = sin(x)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = x.cos();
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ cos(x) dx");
    verify_ftc(&integrand, &result, &x, POINTS_WITH_ZERO, 1e-8, "∫ cos(x) dx");
}

#[test]
fn e2e_exp_x() {
    // ∫ exp(x) dx = exp(x)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = x.exp();
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ exp(x) dx");
    verify_ftc(&integrand, &result, &x, &[-2, -1, 0, 1, 2], 1e-8, "∫ exp(x) dx");
}

#[test]
fn e2e_x_times_exp_x() {
    // ∫ x·exp(x) dx = (x-1)·exp(x)  (by parts)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = &x * &x.exp();
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ x·exp(x) dx");
    verify_ftc(&integrand, &result, &x, &[-2, -1, 0, 1, 2], 1e-8, "∫ x·exp(x) dx");
}

#[test]
fn e2e_ln_x() {
    // ∫ ln(x) dx = x·ln(x) - x  (by parts or Risch log case)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = x.ln();
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ ln(x) dx");
    verify_ftc(&integrand, &result, &x, POSITIVE_POINTS, 1e-8, "∫ ln(x) dx");
}

#[test]
fn e2e_one_over_x_ln_x() {
    // ∫ 1/(x·ln(x)) dx = ln(ln(x))
    //
    // The expression 1/(x·ln(x)) is stored as Pow(Mul(x, Ln(x)), -1).
    // The Pow arm distributes the inverse: Mul(x^(-1), ln(x)^(-1)).
    // The Mul arm's u-sub then finds u = ln(x), du = 1/x dx,
    // giving ∫ 1/u du = ln(u) = ln(ln(x)).
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = &ctx.int(1) / &(&x * &x.ln());
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ 1/(x·ln(x)) dx");
    // Only use x > 1 so ln(x) > 0.
    verify_ftc(&integrand, &result, &x, &[2, 3, 5, 7], 1e-6, "∫ 1/(x·ln(x)) dx");
}

#[test]
fn e2e_x_squared_exp_x() {
    // ∫ x²·exp(x) dx = (x² - 2x + 2)·exp(x)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = &x.powi(2) * &x.exp();
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ x²·exp(x) dx");
    verify_ftc(&integrand, &result, &x, &[-2, -1, 0, 1, 2], 1e-7, "∫ x²·exp(x) dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// Group 4 continued: Definite integrals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn e2e_definite_x_squared_0_to_1() {
    // ∫₀¹ x² dx = 1/3
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = expr!(ctx, x ^ 2).definite_integral(&x, &ctx.int(0), &ctx.int(1));
    let s = format!("{result}");
    assert_eq!(s, "1/3", "∫₀¹ x² dx should be 1/3, got: {s}");
}

#[test]
fn e2e_definite_sin_0_to_pi() {
    // ∫₀^π sin(x) dx = 2
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = x.sin().definite_integral(&x, &ctx.int(0), &ctx.pi());
    let result_eval = result.eval();
    let s = format!("{result_eval}");
    assert_eq!(s, "2", "∫₀^π sin(x) dx should be 2, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Group 6: Regression guards — Risch wiring must not break existing integrals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn risch_doesnt_regress_sin_x() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = x.sin().integrate(&x);
    assert_evaluated(&result, "∫ sin(x) dx regression");
    verify_ftc(&x.sin(), &result, &x, POINTS_WITH_ZERO, 1e-8, "sin(x) regression");
}

#[test]
fn risch_doesnt_regress_cos_x() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = x.cos().integrate(&x);
    assert_evaluated(&result, "∫ cos(x) dx regression");
    verify_ftc(&x.cos(), &result, &x, POINTS_WITH_ZERO, 1e-8, "cos(x) regression");
}

#[test]
fn risch_doesnt_regress_exp_x() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = x.exp().integrate(&x);
    assert_evaluated(&result, "∫ exp(x) dx regression");
    verify_ftc(&x.exp(), &result, &x, &[-2, -1, 0, 1, 2], 1e-8, "exp(x) regression");
}

#[test]
fn risch_doesnt_regress_x_exp_x_by_parts() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = &x * &x.exp();
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ x·exp(x) dx regression");
    verify_ftc(&integrand, &result, &x, &[-2, -1, 0, 1, 2], 1e-8, "x·exp(x) regression");
}

#[test]
fn risch_doesnt_regress_x_sin_x_by_parts() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = &x * &x.sin();
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ x·sin(x) dx regression");
    verify_ftc(&integrand, &result, &x, POINTS_WITH_ZERO, 1e-8, "x·sin(x) regression");
}

#[test]
fn risch_doesnt_regress_polynomial_power_rule() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = expr!(ctx, 5 * x ^ 4 + 3 * x ^ 2 + 1);
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ 5x⁴+3x²+1 dx regression");
    verify_ftc(&integrand, &result, &x, POINTS_WITH_ZERO, 1e-8, "poly regression");
}

#[test]
fn risch_doesnt_regress_trig_identity() {
    // ∫ sin²(x) dx — not affected by Risch (not a rational function)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = x.sin().powi(2);
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ sin²(x) dx regression");
    verify_ftc(&integrand, &result, &x, POINTS_WITH_ZERO, 1e-7, "sin²(x) regression");
}

#[test]
fn risch_doesnt_regress_partial_fractions() {
    // ∫ 1/(x²-1) dx — existing partial fractions AND Risch both handle this.
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = &ctx.int(1) / &expr!(ctx, x ^ 2 - 1);
    let result = integrand.integrate(&x);
    assert_evaluated(&result, "∫ 1/(x²-1) dx regression");
    verify_ftc(&integrand, &result, &x, &[-3, -2, 2, 3, 5], 1e-8, "1/(x²-1) regression");
}

// ═══════════════════════════════════════════════════════════════════════════
// FTC verification for known-answer integrals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ftc_x_to_the_n() {
    // ∫ xⁿ dx for n = 0..6, verify FTC at multiple points.
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    for n in 0..=6 {
        let integrand = x.powi(n);
        let result = integrand.integrate(&x);
        assert_evaluated(&result, &format!("∫ x^{n} dx"));
        verify_ftc(
            &integrand,
            &result,
            &x,
            POINTS_WITH_ZERO,
            1e-8,
            &format!("∫ x^{n} dx"),
        );
    }
}

#[test]
fn ftc_one_over_x_to_the_n() {
    // ∫ x^(-n) dx for n = 1..5, verify FTC.
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    for n in 1..=5 {
        let integrand = x.powi(-n);
        let result = integrand.integrate(&x);
        assert_evaluated(&result, &format!("∫ x^(-{n}) dx"));
        verify_ftc(
            &integrand,
            &result,
            &x,
            SAFE_POINTS,
            1e-7,
            &format!("∫ x^(-{n}) dx"),
        );
    }
}

#[test]
fn ftc_simple_rational_functions() {
    // A suite of simple rational functions, all verified by FTC.
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    let test_cases: Vec<(&str, Ex, &[i64])> = vec![
        ("1/(x+1)", &ctx.int(1) / &(&x + 1), &[-3, -2, 0, 2, 3]),
        ("1/(x+2)", &ctx.int(1) / &(&x + 2), &[-3, -1, 0, 1, 3]),
        ("x/(x+1)", &x / &(&x + 1), &[-3, -2, 0, 2, 3]),
        ("1/(x²+1)", &ctx.int(1) / &(&x.powi(2) + 1), &[-3, -2, 0, 2, 3]),
    ];

    for (label, integrand, points) in &test_cases {
        let result = integrand.integrate(&x);
        assert_evaluated(&result, label);
        verify_ftc(integrand, &result, &x, points, 1e-7, label);
    }
}
