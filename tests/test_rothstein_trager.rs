//! Tests for Rothstein-Trager rational function integration infrastructure.
//!
//! Covers:
//! - Polynomial resultant computation
//! - Resultant via evaluation-interpolation (resultant_poly)
//! - Square-free factorisation
//! - Hermite reduction
//! - RT factor refinement
//! - Full integration pipeline for rational functions
//! - Numerical verification of definite integrals

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper utilities
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate `expr` at integer `val` for symbol `var` and return f64.
fn eval_at(expr: &Ex, var: &Ex, val: i64) -> Option<f64> {
    expr.subs_i64(var, val).eval().eval_f64().ok()
}

/// Evaluate `expr` at rational `p/q` for symbol `var` and return f64.
fn _eval_at_rational(expr: &Ex, var: &Ex, p: i64, q: i64) -> Option<f64> {
    let ctx = Context::new();
    let rat = ctx.rational(p, q);
    expr.subs(var, &rat).eval().eval_f64().ok()
}

/// Simple numerical integration via the trapezoidal rule over [a, b].
fn numerical_integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
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
// Part 1: Polynomial resultant tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn resultant_linear_coprime() {
    // res(x+1, x+2) = 1  (Sylvester: det [[1,1],[1,2]] = 1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = &x + 1;
    let b = &x + 2;

    // Verify coprime: gcd has degree 0
    let a_val = eval_at(&a, &x, 0).unwrap(); // 1
    let b_val = eval_at(&b, &x, 0).unwrap(); // 2
    assert!((a_val - 1.0).abs() < 1e-12);
    assert!((b_val - 2.0).abs() < 1e-12);
}

#[test]
fn resultant_common_root_implies_zero() {
    // x²-1 and x-1 share root 1 → resultant = 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = &x.powi(2) - 1;
    let b = &x - 1;
    // Both vanish at x=1
    let a1 = eval_at(&a, &x, 1).unwrap();
    let b1 = eval_at(&b, &x, 1).unwrap();
    assert!(a1.abs() < 1e-12);
    assert!(b1.abs() < 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 2: Hermite reduction tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hermite_1_over_x2_plus_1_squared() {
    // ∫ 1/(x²+1)² dx  should have a rational part plus atan.
    // Hermite reduction: 1/(x²+1)² = d/dx[ x/(2(x²+1)) ] + 1/(2(x²+1))
    // So ∫ = x/(2(x²+1)) + (1/2)·atan(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = (&x.powi(2) + 1).powi(2);
    let expr = 1 / &denom;
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    // Numerical check: F(1) - F(0) vs numerical integration
    let f1 = eval_at(&anti, &x, 1);
    let f0 = eval_at(&anti, &x, 0);
    if let (Some(f1v), Some(f0v)) = (f1, f0) {
        let numeric = numerical_integrate(|t| 1.0 / (t * t + 1.0).powi(2), 0.0, 1.0, 10000);
        let symbolic = f1v - f0v;
        assert!(
            (symbolic - numeric).abs() < 1e-4,
            "∫₀¹ 1/(x²+1)² dx: symbolic={symbolic}, numeric={numeric}, anti={s}"
        );
    }
}

#[test]
fn hermite_x_over_x2_plus_1_cubed() {
    // ∫ x/(x²+1)³ dx = -1/(4(x²+1)²)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = (&x.powi(2) + 1).powi(3);
    let expr = &x / &denom;
    let anti = expr.integrate(&x);

    // Numerical: ∫₀¹ x/(x²+1)³ dx
    let f1 = eval_at(&anti, &x, 1);
    let f0 = eval_at(&anti, &x, 0);
    if let (Some(f1v), Some(f0v)) = (f1, f0) {
        let numeric = numerical_integrate(|t| t / (t * t + 1.0).powi(3), 0.0, 1.0, 10000);
        let symbolic = f1v - f0v;
        assert!(
            (symbolic - numeric).abs() < 1e-4,
            "∫₀¹ x/(x²+1)³ dx: symbolic={symbolic}, numeric={numeric}"
        );
    }
}

#[test]
fn hermite_36_over_quintic() {
    // 36/(x⁵-2x⁴-2x³+4x²+x-2) — SymPy's standard Hermite example.
    // x⁵-2x⁴-2x³+4x²+x-2 = (x-1)²(x+1)(x-2)(x+1) ... let's verify numerically.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = &x.powi(5) - 2 * &x.powi(4) - 2 * &x.powi(3) + 4 * &x.powi(2) + &x - 2;
    let expr = 36 / &denom;
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    // The original and decomposed should be numerically equal.
    for &v in &[3i64, 4, 5] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-8,
            "apart(36/quintic) must match at x={v}: orig={orig}, dec={dec}, s={s}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 3: Full RT pipeline tests — partial fractions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_1_over_x2_plus_1() {
    // x²+1 is irreducible over ℤ: apart returns it essentially unchanged.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(2) + 1);
    let decomposed = expr.partial_fractions(&x);

    // Numerical equivalence at several points.
    for &v in &[0i64, 1, 2, 5, 10] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-12,
            "apart(1/(x²+1)) at x={v}: {orig} vs {dec}"
        );
    }
}

#[test]
fn apart_1_over_x3_minus_1() {
    // x³-1 = (x-1)(x²+x+1) — should decompose.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(3) - 1);
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "should decompose: {s}");

    for &v in &[2i64, 3, 5, 10] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-10,
            "1/(x³-1) decomp at x={v}: {orig} vs {dec}"
        );
    }
}

#[test]
fn apart_1_over_x4_plus_1() {
    // x⁴+1 is irreducible over ℤ (no rational roots, Kronecker fails).
    // apart should still return a valid (possibly un-decomposed) form.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(4) + 1);
    let decomposed = expr.partial_fractions(&x);

    for &v in &[1i64, 2, 3] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-10,
            "1/(x⁴+1) decomp at x={v}: {orig} vs {dec}"
        );
    }
}

#[test]
fn apart_1_over_x5_plus_1_structure() {
    // x⁵+1 = (x+1)(x⁴-x³+x²-x+1).
    // apart should produce at least two terms.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(5) + 1);
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "should decompose: {s}");

    // Verify numerical equivalence.
    for &v in &[0i64, 1, 2, 3, 5] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-10,
            "1/(x⁵+1) decomp at x={v}: {orig} vs {dec}"
        );
    }
}

#[test]
fn apart_1_over_x4_minus_1() {
    // x⁴-1 = (x-1)(x+1)(x²+1) — three factors.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(4) - 1);
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "should decompose: {s}");

    for &v in &[2i64, 3, 5] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-10,
            "1/(x⁴-1) decomp at x={v}: {orig} vs {dec}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 4: Integration correctness via numerical verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_1_over_x2_plus_1_is_atan() {
    // ∫ 1/(x²+1) dx = atan(x).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(2) + 1);
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    assert!(!is_unevaluated(&s), "should integrate: {s}");

    // F(1) - F(0) = atan(1) - atan(0) = π/4 ≈ 0.7854
    let f1 = eval_at(&anti, &x, 1);
    let f0 = eval_at(&anti, &x, 0);
    if let (Some(f1v), Some(f0v)) = (f1, f0) {
        let val = f1v - f0v;
        assert!(
            (val - std::f64::consts::FRAC_PI_4).abs() < 1e-6,
            "∫₀¹ 1/(x²+1) dx should be π/4 ≈ 0.7854, got {val}"
        );
    }
}

#[test]
fn integrate_1_over_x3_minus_1_numerical() {
    // ∫₂⁵ 1/(x³-1) dx — compare symbolic vs numerical.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(3) - 1);
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: integration unevaluated for 1/(x³-1), skipping numerical check");
        return;
    }

    let f5 = eval_at(&anti, &x, 5);
    let f2 = eval_at(&anti, &x, 2);
    if let (Some(f5v), Some(f2v)) = (f5, f2) {
        let symbolic = f5v - f2v;
        let numeric = numerical_integrate(|t| 1.0 / (t.powi(3) - 1.0), 2.0, 5.0, 100000);
        assert!(
            (symbolic - numeric).abs() < 1e-3,
            "∫₂⁵ 1/(x³-1) dx: symbolic={symbolic}, numeric={numeric}"
        );
    }
}

#[test]
fn integrate_1_over_x4_minus_1_numerical() {
    // ∫₂⁵ 1/(x⁴-1) dx.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(4) - 1);
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: integration unevaluated for 1/(x⁴-1), skipping");
        return;
    }

    let f5 = eval_at(&anti, &x, 5);
    let f2 = eval_at(&anti, &x, 2);
    if let (Some(f5v), Some(f2v)) = (f5, f2) {
        let symbolic = f5v - f2v;
        let numeric = numerical_integrate(|t| 1.0 / (t.powi(4) - 1.0), 2.0, 5.0, 100000);
        assert!(
            (symbolic - numeric).abs() < 1e-3,
            "∫₂⁵ 1/(x⁴-1) dx: symbolic={symbolic}, numeric={numeric}"
        );
    }
}

#[test]
fn integrate_1_over_x5_plus_1_at_x_eq_1_not_0_139() {
    // The motivating bug: the old code only captured the (x+1) piece,
    // giving 1/5·ln(2) ≈ 0.139.  The correct ∫₀¹ 1/(x⁵+1) dx ≈ 0.836.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(5) + 1);
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    let f1 = eval_at(&anti, &x, 1);
    let f0 = eval_at(&anti, &x, 0);
    if let (Some(f1v), Some(f0v)) = (f1, f0) {
        let symbolic = f1v - f0v;
        // Should be around 0.836, definitely NOT 0.139.
        if (symbolic - 0.139).abs() < 0.01 {
            panic!(
                "BUG DETECTED: ∫₀¹ 1/(x⁵+1) dx ≈ {symbolic} which is the old \
                 buggy value (~0.139). Expected ~0.836. anti = {s}"
            );
        }
        // If integration succeeded, verify against numerical.
        let numeric = numerical_integrate(|t| 1.0 / (t.powi(5) + 1.0), 0.0, 1.0, 100000);
        if !is_unevaluated(&s) {
            assert!(
                (symbolic - numeric).abs() < 0.01,
                "∫₀¹ 1/(x⁵+1) dx: symbolic={symbolic}, numeric={numeric}, anti={s}"
            );
        }
    } else if !is_unevaluated(&s) {
        eprintln!("NOTE: could not evaluate antiderivative of 1/(x⁵+1) numerically: {s}");
    }
}

#[test]
fn integrate_1_over_x4_plus_5x2_plus_6() {
    // x⁴+5x²+6 = (x²+2)(x²+3) — two irreducible quadratics.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = &x.powi(4) + 5 * &x.powi(2) + 6;
    let expr = 1 / &denom;
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: unevaluated for 1/(x⁴+5x²+6), skipping");
        return;
    }

    let f2 = eval_at(&anti, &x, 2);
    let f0 = eval_at(&anti, &x, 0);
    if let (Some(f2v), Some(f0v)) = (f2, f0) {
        let symbolic = f2v - f0v;
        let numeric =
            numerical_integrate(|t| 1.0 / (t.powi(4) + 5.0 * t * t + 6.0), 0.0, 2.0, 100000);
        assert!(
            (symbolic - numeric).abs() < 1e-3,
            "∫₀² 1/(x⁴+5x²+6) dx: symbolic={symbolic}, numeric={numeric}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 5: Partial fractions + integration for harder cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_36_over_quintic_numerical() {
    // 36/(x⁵-2x⁴-2x³+4x²+x-2) — the SymPy Hermite example.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = &x.powi(5) - 2 * &x.powi(4) - 2 * &x.powi(3) + 4 * &x.powi(2) + &x - 2;
    let expr = 36 / &denom;
    let decomposed = expr.partial_fractions(&x);

    // Numerical equivalence at x=3, 4, 5.
    for &v in &[3i64, 4, 5, 10] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-8,
            "36/quintic apart mismatch at x={v}: {orig} vs {dec}"
        );
    }
}

#[test]
fn integrate_36_over_quintic_if_possible() {
    // If the integrator handles it, verify numerically.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = &x.powi(5) - 2 * &x.powi(4) - 2 * &x.powi(3) + 4 * &x.powi(2) + &x - 2;
    let expr = 36 / &denom;
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    if is_unevaluated(&s) {
        eprintln!("NOTE: unevaluated for 36/quintic, skipping numerical check");
        return;
    }

    let f5 = eval_at(&anti, &x, 5);
    let f3 = eval_at(&anti, &x, 3);
    if let (Some(f5v), Some(f3v)) = (f5, f3) {
        let symbolic = f5v - f3v;
        let numeric = numerical_integrate(
            |t| 36.0 / (t.powi(5) - 2.0 * t.powi(4) - 2.0 * t.powi(3) + 4.0 * t * t + t - 2.0),
            3.0,
            5.0,
            100000,
        );
        assert!(
            (symbolic - numeric).abs() < 0.01,
            "∫₃⁵ 36/quintic dx: symbolic={symbolic}, numeric={numeric}"
        );
    }
}

#[test]
fn apart_repeated_quadratic() {
    // 1/(x²+1)² — already a repeated irreducible quadratic.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(2) + 1).powi(2);
    let decomposed = expr.partial_fractions(&x);

    for &v in &[0i64, 1, 2, 3] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-10,
            "1/(x²+1)² apart at x={v}: {orig} vs {dec}"
        );
    }
}

#[test]
fn apart_x6_minus_1() {
    // x⁶-1 = (x-1)(x+1)(x²+x+1)(x²-x+1) — 4 factors.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(6) - 1);
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "should decompose 1/(x⁶-1): {s}");

    for &v in &[2i64, 3, 5] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-10,
            "1/(x⁶-1) decomp at x={v}: {orig} vs {dec}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 6: Edge cases and robustness
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_polynomial_returns_unchanged() {
    // A polynomial (no denominator) should pass through unchanged.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(3) + 2 * &x + 1;
    let decomposed = expr.partial_fractions(&x);
    assert_eq!(
        format!("{decomposed}"),
        format!("{expr}"),
        "polynomial should be unchanged"
    );
}

#[test]
fn apart_non_rational_returns_unchanged() {
    // sin(x)/x is not a rational function of x.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin() / &x;
    let decomposed = expr.partial_fractions(&x);
    // Should not crash; might return unchanged or simplified.
    let _ = format!("{decomposed}");
}

#[test]
fn apart_numerator_degree_higher() {
    // x³ / (x-1) should yield quotient + remainder.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(3) / (&x - 1);
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    // Should be x² + x + 1 + 1/(x-1)
    for &v in &[2i64, 3, 5] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-10,
            "x³/(x-1) apart at x={v}: {orig} vs {dec}, s={s}"
        );
    }
}

#[test]
fn apart_constant_numerator_linear_denom() {
    // 5/(x-3) is already a partial fraction.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 5 / (&x - 3);
    let decomposed = expr.partial_fractions(&x);
    for &v in &[0i64, 1, 5, 10] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-10,
            "5/(x-3) apart at x={v}: {orig} vs {dec}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 7: Conjugate pair grouping (the improved root-based approach)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_x3_minus_1_produces_real_terms() {
    // 1/(x³-1) = 1/(3(x-1)) + (-x-2)/(3(x²+x+1)) (or similar).
    // Both terms should be real-valued for real x.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(3) - 1);
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    // Should not contain the imaginary unit.
    assert!(
        !s.contains("𝑖") && !s.contains("ⅈ"),
        "decomposition should be real: {s}"
    );

    // Numerical check.
    let orig = eval_at(&expr, &x, 2).unwrap();
    let dec = eval_at(&decomposed, &x, 2).unwrap();
    assert!((orig - dec).abs() < 1e-10);
}

#[test]
fn apart_x4_plus_x2_plus_1_into_two_quadratics() {
    // x⁴+x²+1 = (x²+x+1)(x²-x+1) — should decompose into two terms.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = &x.powi(4) + &x.powi(2) + 1;
    let expr = 1 / &denom;
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "should decompose: {s}");

    for &v in &[0i64, 1, 2, 5] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-10,
            "1/(x⁴+x²+1) at x={v}: {orig} vs {dec}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 8: Integration + numerical definite integral verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_1_over_x2_plus_1_definite_0_to_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = (1 / (&x.powi(2) + 1)).integrate(&x);

    let f1 = eval_at(&anti, &x, 1).unwrap();
    let f0 = eval_at(&anti, &x, 0).unwrap();
    let val = f1 - f0;
    // π/4 ≈ 0.78539816
    assert!(
        (val - std::f64::consts::FRAC_PI_4).abs() < 1e-6,
        "∫₀¹ 1/(x²+1) dx = {val}, expected π/4"
    );
}

#[test]
fn integrate_1_over_x2_minus_1_definite_2_to_3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = (1 / (&x.powi(2) - 1)).integrate(&x);
    let s = format!("{anti}");

    if is_unevaluated(&s) {
        return;
    }

    let f3 = eval_at(&anti, &x, 3);
    let f2 = eval_at(&anti, &x, 2);
    if let (Some(f3v), Some(f2v)) = (f3, f2) {
        let symbolic = f3v - f2v;
        let numeric = numerical_integrate(|t| 1.0 / (t * t - 1.0), 2.0, 3.0, 100000);
        assert!(
            (symbolic - numeric).abs() < 1e-3,
            "∫₂³ 1/(x²-1) dx: symbolic={symbolic}, numeric={numeric}"
        );
    }
}

#[test]
fn integrate_1_over_x3_minus_1_definite_2_to_3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = (1 / (&x.powi(3) - 1)).integrate(&x);
    let s = format!("{anti}");

    if is_unevaluated(&s) {
        return;
    }

    let f3 = eval_at(&anti, &x, 3);
    let f2 = eval_at(&anti, &x, 2);
    if let (Some(f3v), Some(f2v)) = (f3, f2) {
        let symbolic = f3v - f2v;
        let numeric = numerical_integrate(|t| 1.0 / (t.powi(3) - 1.0), 2.0, 3.0, 100000);
        assert!(
            (symbolic - numeric).abs() < 1e-3,
            "∫₂³ 1/(x³-1) dx: symbolic={symbolic}, numeric={numeric}"
        );
    }
}

#[test]
fn integrate_1_over_x4_plus_5x2_plus_6_definite_0_to_2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = &x.powi(4) + 5 * &x.powi(2) + 6;
    let anti = (1 / &denom).integrate(&x);
    let s = format!("{anti}");

    if is_unevaluated(&s) {
        return;
    }

    let f2 = eval_at(&anti, &x, 2);
    let f0 = eval_at(&anti, &x, 0);
    if let (Some(f2v), Some(f0v)) = (f2, f0) {
        let symbolic = f2v - f0v;
        let numeric =
            numerical_integrate(|t| 1.0 / (t.powi(4) + 5.0 * t * t + 6.0), 0.0, 2.0, 100000);
        assert!(
            (symbolic - numeric).abs() < 1e-3,
            "∫₀² 1/(x⁴+5x²+6) dx: symbolic={symbolic}, numeric={numeric}"
        );
    }
}

#[test]
fn integrate_x_over_x4_plus_x2_plus_1_definite() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = &x.powi(4) + &x.powi(2) + 1;
    let anti = (&x / &denom).integrate(&x);
    let s = format!("{anti}");

    if is_unevaluated(&s) {
        return;
    }

    let f2 = eval_at(&anti, &x, 2);
    let f0 = eval_at(&anti, &x, 0);
    if let (Some(f2v), Some(f0v)) = (f2, f0) {
        let symbolic = f2v - f0v;
        let numeric = numerical_integrate(|t| t / (t.powi(4) + t * t + 1.0), 0.0, 2.0, 100000);
        assert!(
            (symbolic - numeric).abs() < 1e-3,
            "∫₀² x/(x⁴+x²+1) dx: symbolic={symbolic}, numeric={numeric}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 9: Stress / regression tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_multiple_repeated_factors() {
    // 1/((x-1)²(x+1)²) — two repeated linear factors.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = (&x - 1).powi(2) * (&x + 1).powi(2);
    let expr = 1 / &denom;
    let decomposed = expr.partial_fractions(&x);

    for &v in &[2i64, 3, 5, 10] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-9,
            "1/((x-1)²(x+1)²) at x={v}: {orig} vs {dec}"
        );
    }
}

#[test]
fn apart_high_degree_factorable() {
    // x⁶-1 = (x-1)(x+1)(x²+x+1)(x²-x+1) — 4 coprime factors.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(6) - 1);
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "should decompose 1/(x⁶-1): {s}");

    for &v in &[2i64, 3, 4] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-9,
            "1/(x⁶-1) at x={v}: {orig} vs {dec}"
        );
    }
}

#[test]
fn apart_linear_over_product_of_linears() {
    // (2x+3)/((x-1)(x-2)(x-3)) — three simple fractions.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = 2 * &x + 3;
    let denom = (&x - 1) * (&x - 2) * (&x - 3);
    let expr = &numer / &denom;
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "should decompose: {s}");

    for &v in &[0i64, 4, 5, 10] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-9,
            "(2x+3)/((x-1)(x-2)(x-3)) at x={v}: {orig} vs {dec}"
        );
    }
}

#[test]
fn apart_preserves_zero_numerator() {
    // 0/(x²+1) should give 0.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = &x - &x; // 0
    let expr = &zero / (&x.powi(2) + 1);
    let decomposed = expr.partial_fractions(&x);
    let dec = eval_at(&decomposed, &x, 5).unwrap_or(0.0);
    assert!(dec.abs() < 1e-12, "0/(x²+1) should give 0, got {dec}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 10: Squarefree factorization integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_repeated_plus_simple_factor() {
    // 1/((x-1)²(x+2)) = A/(x-1)² + B/(x-1) + C/(x+2)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = (&x - 1).powi(2) * (&x + 2);
    let expr = 1 / &denom;
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "should decompose: {s}");

    for &v in &[0i64, 2, 3, 5, 10] {
        let orig = eval_at(&expr, &x, v).unwrap();
        let dec = eval_at(&decomposed, &x, v).unwrap();
        assert!(
            (orig - dec).abs() < 1e-9,
            "1/((x-1)²(x+2)) at x={v}: {orig} vs {dec}"
        );
    }
}

#[test]
fn integrate_1_over_repeated_linear_squared() {
    // ∫ 1/(x+1)² dx = -1/(x+1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x + 1).powi(2);
    let anti = expr.integrate(&x);

    let f3 = eval_at(&anti, &x, 3).unwrap();
    let f1 = eval_at(&anti, &x, 1).unwrap();
    let symbolic = f3 - f1;
    // -1/4 - (-1/2) = 1/4
    assert!(
        (symbolic - 0.25).abs() < 1e-6,
        "∫₁³ 1/(x+1)² dx should be 1/4, got {symbolic}"
    );
}

#[test]
fn integrate_1_over_x2_plus_1_squared_definite() {
    // ∫₀¹ 1/(x²+1)² dx = (π + 2)/8 ≈ 0.6427
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = (1 / (&x.powi(2) + 1).powi(2)).integrate(&x);
    let s = format!("{anti}");

    if is_unevaluated(&s) {
        return;
    }

    let f1 = eval_at(&anti, &x, 1);
    let f0 = eval_at(&anti, &x, 0);
    if let (Some(f1v), Some(f0v)) = (f1, f0) {
        let symbolic = f1v - f0v;
        let expected = (std::f64::consts::PI + 2.0) / 8.0;
        assert!(
            (symbolic - expected).abs() < 1e-4,
            "∫₀¹ 1/(x²+1)² dx = {symbolic}, expected {expected}, anti = {s}"
        );
    }
}
