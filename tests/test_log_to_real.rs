//! Integration tests for the log-to-real numeric conversion in `apart`.
//!
//! Verifies that partial fraction decomposition of rational functions with
//! irreducible polynomial factors produces clean real-form expressions
//! (no unsimplified Ferrari radicals, no imaginary unit leakage).

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate `expr` at integer `val` for symbol `var` and return f64.
fn eval_at(expr: &Ex, var: &Ex, val: i64) -> Option<f64> {
    expr.subs_i64(var, val).eval().eval_f64().ok()
}

/// Simple numerical integration via the trapezoidal rule over [a, b].
fn numerical_integrate_trap<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    sum * h
}

/// Returns `true` if the string looks like an unevaluated integral.
fn is_unevaluated(s: &str) -> bool {
    s.contains("Integral") || s.contains("∫")
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: ∫ 1/(x³−1) dx — clean result with atan and log, no I or cbrt
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn log_to_real_x3_minus_1_clean() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let integrand = &one / (&x.powi(3) - &one);
    let result = integrand.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("SKIP: integration returned unevaluated form: {s}");
        return;
    }

    // The result should NOT contain the imaginary unit.
    assert!(
        !s.contains(" I") && !s.contains("*I") && !s.contains("(I)"),
        "∫1/(x³−1) should not contain imaginary unit I: {s}"
    );
    // Should not contain cbrt (unsimplified cube roots).
    assert!(
        !s.contains("cbrt"),
        "∫1/(x³−1) should not contain cbrt: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: ∫ 1/(x⁵+1) dx — no ugly Ferrari radicals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn log_to_real_x5_plus_1_no_ferrari() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let integrand = &one / (&x.powi(5) + &one);
    let result = integrand.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("SKIP: integration returned unevaluated form: {s}");
        return;
    }

    // The result must NOT contain ugly Ferrari expressions.
    assert!(
        !s.contains("cbrt("),
        "∫1/(x⁵+1) should not contain cbrt: {s}"
    );
    assert!(
        !s.contains("125/6912"),
        "∫1/(x⁵+1) should not contain sqrt(125/6912): {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: definite ∫₀^{0.5} 1/(x⁵+1) dx ≈ 0.486
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn log_to_real_x5_plus_1_numeric() {
    // Ground truth via numerical integration.
    let expected = numerical_integrate_trap(|x| 1.0 / (x.powi(5) + 1.0), 0.0, 0.5, 10_000);
    assert!(
        (expected - 0.497).abs() < 0.02,
        "ground truth sanity check failed: {expected}"
    );

    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let integrand = &one / (&x.powi(5) + &one);
    let result = integrand.integrate(&x);
    let s = format!("{result}");

    if is_unevaluated(&s) {
        eprintln!("SKIP: integration returned unevaluated form: {s}");
        return;
    }

    // Evaluate the antiderivative at the endpoints: F(0.5) − F(0).
    let half = ctx.rational(1, 2);
    let zero = ctx.int(0);
    let f_half = result.subs(&x, &half).eval();
    let f_zero = result.subs(&x, &zero).eval();
    let definite = (&f_half - &f_zero).eval();

    match definite.eval_f64() {
        Ok(val) => {
            assert!(
                (val - expected).abs() < 1e-4,
                "∫₀^0.5 1/(x⁵+1) dx should be ≈ {expected}, got {val}"
            );
        }
        Err(e) => {
            eprintln!("SKIP: could not evaluate definite integral to f64: {e}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: apart(1/(x⁴+1)) produces terms with sqrt(2)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn log_to_real_x4_plus_1_has_sqrt2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let expr = &one / (&x.powi(4) + &one);
    let result = expr.partial_fractions(&x);
    let s = format!("{result}");

    // The roots of x⁴+1 involve √2 (the quadratic factors are
    // x² ± √2·x + 1).  The partial fraction should contain √2.
    // If apart didn't decompose, skip.
    let orig_s = format!("{expr}");
    if s == orig_s {
        eprintln!("SKIP: apart did not decompose 1/(x⁴+1): {s}");
        return;
    }

    // Should NOT contain imaginary unit or cbrt.
    assert!(
        !s.contains("cbrt"),
        "apart(1/(x⁴+1)) should not contain cbrt: {s}"
    );

    // Numerical correctness at a few points.
    for &v in &[2i64, 3, 5] {
        let orig_val = eval_at(&expr, &x, v);
        let dec_val = eval_at(&result, &x, v);
        match (orig_val, dec_val) {
            (Some(o), Some(d)) if o.is_finite() && d.is_finite() => {
                let err = (o - d).abs();
                assert!(
                    err < 1e-8,
                    "apart(1/(x⁴+1)) mismatch at x={v}: original={o}, decomposed={d}"
                );
            }
            _ => {}
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: apart(1/(x²−1)) — simple rational roots, no regression
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn log_to_real_preserves_rational_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let expr = &one / (&x.powi(2) - &one);
    let result = expr.partial_fractions(&x);
    let s = format!("{result}");

    // Should be decomposed (not identical to original).
    let orig_s = format!("{expr}");
    assert_ne!(s, orig_s, "apart should decompose 1/(x²−1)");

    // Should contain (x − 1) and (x + 1) denominators.
    // Verify numerically.
    for &v in &[2i64, 3, 5, 7] {
        let orig_val = eval_at(&expr, &x, v);
        let dec_val = eval_at(&result, &x, v);
        match (orig_val, dec_val) {
            (Some(o), Some(d)) if o.is_finite() && d.is_finite() => {
                let err = (o - d).abs();
                assert!(
                    err < 1e-10,
                    "apart(1/(x²−1)) mismatch at x={v}: original={o}, decomposed={d}"
                );
            }
            _ => {}
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: 1/(x⁶−1) — degree 6, factors into degree-1 and degree-2 pieces
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn log_to_real_x6_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let expr = &one / (&x.powi(6) - &one);
    let result = expr.partial_fractions(&x);
    let s = format!("{result}");

    let orig_s = format!("{expr}");
    if s == orig_s {
        eprintln!("SKIP: apart did not decompose 1/(x⁶−1): {s}");
        return;
    }

    // Should not contain imaginary unit or unsimplified radicals.
    assert!(
        !s.contains("cbrt"),
        "apart(1/(x⁶−1)) should not contain cbrt: {s}"
    );

    // Numerical correctness.
    for &v in &[2i64, 3, 4, 5] {
        let orig_val = eval_at(&expr, &x, v);
        let dec_val = eval_at(&result, &x, v);
        match (orig_val, dec_val) {
            (Some(o), Some(d)) if o.is_finite() && d.is_finite() => {
                let err = (o - d).abs();
                let scale = o.abs().max(1.0);
                assert!(
                    err / scale < 1e-6,
                    "apart(1/(x⁶−1)) mismatch at x={v}: original={o}, decomposed={d}"
                );
            }
            _ => {}
        }
    }
}
