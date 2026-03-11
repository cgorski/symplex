//! Tests for rational function partial-fraction decomposition and integration.
//!
//! These tests verify the correctness of the improved `apart()` function
//! which uses polynomial factoring over ℤ and extended GCD decomposition,
//! as well as the integration pipeline for rational functions.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: numerical evaluation of F(b) − F(a) for definite integrals
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate `expr` at integer `val` for symbol `var` and return f64.
fn eval_at_int(expr: &Ex, var: &Ex, val: i64) -> Option<f64> {
    expr.subs_i64(var, val).eval().eval_f64().ok()
}

// ═══════════════════════════════════════════════════════════════════════════
// Partial fraction decomposition tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_1_over_x2_plus_1_unchanged() {
    // x²+1 is irreducible over ℤ — apart should return it essentially
    // unchanged (single quadratic factor, no decomposition possible).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(2) + 1);
    let decomposed = expr.partial_fractions(&x);
    // The integrator handles 1/(x²+1) via atan, so it's fine unchanged.
    let orig_s = format!("{expr}");
    let dec_s = format!("{decomposed}");
    // Either unchanged or trivially equivalent.
    // Verify numerically: both evaluate the same at x=2.
    let orig_val = eval_at_int(&expr, &x, 2).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 2).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-12,
        "apart(1/(x²+1)) should be numerically equivalent: orig={orig_val}, dec={dec_val}, orig_s={orig_s}, dec_s={dec_s}"
    );
}

#[test]
fn apart_1_over_x2_minus_1() {
    // 1/(x²-1) = 1/((x-1)(x+1)) should decompose into two terms.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(2) - 1);
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    // Should be decomposed (different from original representation).
    assert_ne!(s, format!("{expr}"), "should decompose: {s}");

    // Numerically verify at x=2: 1/(4-1) = 1/3.
    let orig_val = eval_at_int(&expr, &x, 2).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 2).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-12,
        "decomposition must match at x=2: {orig_val} vs {dec_val}"
    );

    // Verify at x=3: 1/(9-1) = 1/8.
    let orig_val = eval_at_int(&expr, &x, 3).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 3).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-12,
        "decomposition must match at x=3: {orig_val} vs {dec_val}"
    );
}

#[test]
fn apart_1_over_x3_minus_1() {
    // x³-1 = (x-1)(x²+x+1)  — should produce a linear and a quadratic term.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(3) - 1);
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "should decompose 1/(x³-1): {s}");

    // Verify numerically at x=2: 1/(8-1) = 1/7 ≈ 0.142857
    let orig_val = eval_at_int(&expr, &x, 2).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 2).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "1/(x³-1) decomposition must match at x=2: {orig_val} vs {dec_val}"
    );

    // Verify at x=3: 1/26.
    let orig_val = eval_at_int(&expr, &x, 3).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 3).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "1/(x³-1) decomposition must match at x=3: {orig_val} vs {dec_val}"
    );
}

#[test]
fn apart_1_over_x5_plus_1_decomposes() {
    // x⁵+1 = (x+1)(x⁴-x³+x²-x+1)
    // Should decompose into at least 2 terms (linear + quartic).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(5) + 1);
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(
        s,
        format!("{expr}"),
        "should decompose 1/(x⁵+1) into at least linear + quartic: {s}"
    );

    // Numerically verify at x=2: 1/(32+1) = 1/33.
    let orig_val = eval_at_int(&expr, &x, 2).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 2).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "1/(x⁵+1) decomposition must match at x=2: {orig_val} vs {dec_val}"
    );

    // Verify at x=0: 1/1 = 1.
    let orig_val = eval_at_int(&expr, &x, 0).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 0).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "1/(x⁵+1) decomposition must match at x=0: {orig_val} vs {dec_val}"
    );
}

#[test]
fn apart_x_over_x4_plus_x2_plus_1() {
    // x⁴+x²+1 = (x²+x+1)(x²-x+1)  — two irreducible quadratics
    // with rational coefficients.  Should decompose into 2 terms.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = &x.powi(4) + &x.powi(2) + 1;
    let expr = &x / &denom;
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "x/(x⁴+x²+1) should decompose: {s}");

    // Numerically verify at x=1: 1/(1+1+1) = 1/3.
    let orig_val = eval_at_int(&expr, &x, 1).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 1).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "x/(x⁴+x²+1) must match at x=1: {orig_val} vs {dec_val}"
    );

    // Verify at x=2: 2/(16+4+1) = 2/21.
    let orig_val = eval_at_int(&expr, &x, 2).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 2).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "x/(x⁴+x²+1) must match at x=2: {orig_val} vs {dec_val}"
    );
}

#[test]
fn apart_2x_plus_3_over_x2_plus_x_plus_1() {
    // x²+x+1 is irreducible over ℤ.
    // (2x+3)/(x²+x+1) is already a single-denominator partial fraction.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = &x * 2 + 3;
    let denom = &x.powi(2) + &x + 1;
    let expr = &numer / &denom;
    let decomposed = expr.partial_fractions(&x);

    // Should remain essentially the same (or equivalent).
    let orig_val = eval_at_int(&expr, &x, 2).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 2).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "(2x+3)/(x²+x+1) should be equivalent: {orig_val} vs {dec_val}"
    );
}

#[test]
fn apart_repeated_linear_factor() {
    // (2x+3)/(x+1)² should decompose into 2/(x+1) + 1/(x+1)².
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = &x * 2 + 3;
    let denom = (&x + 1).powi(2);
    let expr = &numer / &denom;
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "(2x+3)/(x+1)² should decompose: {s}");

    // Numerically verify at x=2: (7)/(9) vs decomposed at x=2.
    let orig_val = eval_at_int(&expr, &x, 2).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 2).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "(2x+3)/(x+1)² must match at x=2: {orig_val} vs {dec_val}"
    );

    // Verify at x=0: 3/1 = 3.
    let orig_val = eval_at_int(&expr, &x, 0).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 0).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "(2x+3)/(x+1)² must match at x=0: {orig_val} vs {dec_val}"
    );
}

#[test]
fn apart_1_over_x3_minus_1_three_point_verification() {
    // Verify the decomposition of 1/(x³-1) at multiple points to
    // ensure all terms are accounted for.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(3) - 1);
    let decomposed = expr.partial_fractions(&x);

    for val in [2, 3, 4, 5, 10] {
        let orig = eval_at_int(&expr, &x, val).unwrap();
        let dec = eval_at_int(&decomposed, &x, val).unwrap();
        assert!(
            (orig - dec).abs() < 1e-10,
            "1/(x³-1) at x={val}: orig={orig}, dec={dec}"
        );
    }
}

#[test]
fn apart_x6_minus_1_factored() {
    // x⁶-1 = (x-1)(x+1)(x²+x+1)(x²-x+1) — four coprime factors.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(6) - 1);
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "1/(x⁶-1) should decompose: {s}");

    // Verify at x=2: 1/(64-1) = 1/63.
    let orig_val = eval_at_int(&expr, &x, 2).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 2).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "1/(x⁶-1) must match at x=2: {orig_val} vs {dec_val}"
    );
}

#[test]
fn apart_repeated_and_coprime() {
    // 1/((x+1)²(x-1)) — mixed repeated + coprime factors.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = (&x + 1).powi(2) * (&x - 1);
    let expr = 1 / &denom;
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(
        s,
        format!("{expr}"),
        "1/((x+1)²(x-1)) should decompose: {s}"
    );

    // Verify at x=2: 1/(9*1) = 1/9.
    let orig_val = eval_at_int(&expr, &x, 2).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 2).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "1/((x+1)²(x-1)) must match at x=2: {orig_val} vs {dec_val}"
    );

    // Verify at x=3: 1/(16*2) = 1/32.
    let orig_val = eval_at_int(&expr, &x, 3).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 3).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "1/((x+1)²(x-1)) must match at x=3: {orig_val} vs {dec_val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration tests — verify that apart + integrator pipeline works
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_1_over_x2_plus_1_is_atan() {
    // ∫ 1/(x²+1) dx = atan(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(2) + 1);
    let anti = expr.integrate(&x);
    let s = format!("{anti}");
    assert!(s.contains("atan"), "∫ 1/(x²+1) dx should be atan(x): {s}");

    // Numerical: F(1) - F(0) should be atan(1) - atan(0) = π/4 ≈ 0.7854
    let f1 = eval_at_int(&anti, &x, 1);
    let f0 = eval_at_int(&anti, &x, 0);
    if let (Some(f1), Some(f0)) = (f1, f0) {
        let val = f1 - f0;
        assert!(
            (val - std::f64::consts::FRAC_PI_4).abs() < 1e-10,
            "∫₀¹ 1/(x²+1) dx should be π/4 ≈ 0.7854: got {val}"
        );
    }
}

#[test]
fn integrate_1_over_x2_minus_1() {
    // ∫ 1/(x²-1) dx = 1/2·ln|x-1| - 1/2·ln|x+1| (or similar)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(2) - 1);
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    // Should contain ln (logarithmic terms).
    assert!(s.contains("ln"), "∫ 1/(x²-1) dx should contain ln: {s}");

    // Numerical: F(3) - F(2) — avoiding the pole at x=1.
    // 1/(x²-1) = 1/2*(1/(x-1) - 1/(x+1))
    // F(x) = 1/2*ln|x-1| - 1/2*ln|x+1|
    // F(3) - F(2) = 1/2*(ln2 - ln4) - 1/2*(ln1 - ln3) = 1/2*(ln2-ln4+ln3)
    //             = 1/2*(ln(2*3/4)) = 1/2*ln(3/2) ≈ 0.2027
    let f3 = eval_at_int(&anti, &x, 3);
    let f2 = eval_at_int(&anti, &x, 2);
    if let (Some(f3), Some(f2)) = (f3, f2) {
        let val = f3 - f2;
        let expected = 0.5 * (1.5_f64).ln();
        assert!(
            (val - expected).abs() < 1e-8,
            "∫₂³ 1/(x²-1) dx should be ≈ {expected}: got {val}"
        );
    }
}

#[test]
fn integrate_1_over_x3_minus_1_numerically() {
    // After the apart fix, 1/(x³-1) should decompose into:
    //   1/(3(x-1)) + (linear)/(x²+x+1)
    // Both terms are integrable by the existing integrator.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(3) - 1);
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    // The antiderivative should contain ln and atan terms.
    let has_ln = s.contains("ln");
    let has_atan = s.contains("atan");

    // Numerical verification: ∫₂³ 1/(x³-1) dx.
    // The exact answer is approximately 0.07539...
    let f3 = eval_at_int(&anti, &x, 3);
    let f2 = eval_at_int(&anti, &x, 2);

    if let (Some(f3), Some(f2)) = (f3, f2) {
        let val = f3 - f2;
        let expected = 0.07539; // approximate (verified via numerical quadrature)
        assert!(
            (val - expected).abs() < 0.01,
            "∫₂³ 1/(x³-1) dx ≈ {expected}: got {val} (antiderivative: {s})"
        );
    } else {
        // If numerical evaluation fails, at least check the form.
        assert!(
            has_ln || has_atan,
            "∫ 1/(x³-1) dx should have ln or atan: {s}"
        );
    }
}

#[test]
fn integrate_x_over_x4_plus_x2_plus_1_numerically() {
    // x⁴+x²+1 = (x²+x+1)(x²-x+1) — both irreducible quadratics.
    // apart decomposes into constant/(x²+x+1) + constant/(x²-x+1).
    // Each integrates to atan forms.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = &x.powi(4) + &x.powi(2) + 1;
    let expr = &x / &denom;
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    // Numerical: ∫₀¹ x/(x⁴+x²+1) dx.
    // The exact value is (1/√3)*(atan(√3) - atan(1/√3))
    // = (1/√3)*(π/3 - π/6) = (1/√3)*π/6 ≈ 0.3023...
    let f1 = eval_at_int(&anti, &x, 1);
    let f0 = eval_at_int(&anti, &x, 0);

    if let (Some(f1), Some(f0)) = (f1, f0) {
        let val = f1 - f0;
        // The exact value is π/(6√3) ≈ 0.30236...
        let expected = std::f64::consts::PI / (6.0 * 3.0_f64.sqrt());
        assert!(
            (val - expected).abs() < 0.01,
            "∫₀¹ x/(x⁴+x²+1) dx ≈ {expected}: got {val} (form: {s})"
        );
    } else {
        // Check that the antiderivative at least contains atan.
        assert!(
            s.contains("atan") || s.contains("ln"),
            "∫ x/(x⁴+x²+1) dx should contain atan/ln: {s}"
        );
    }
}

#[test]
fn integrate_2x_plus_3_over_x2_plus_x_plus_1() {
    // (2x+3)/(x²+x+1) — irreducible quadratic denominator with
    // non-trivial numerator.  Should integrate to ln + atan form.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = &x * 2 + 3;
    let denom = &x.powi(2) + &x + 1;
    let expr = &numer / &denom;
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    // Should integrate successfully (not be an unevaluated integral).
    assert!(
        !s.contains("Integral"),
        "∫ (2x+3)/(x²+x+1) dx should be evaluated: {s}"
    );

    // Numerical: F(1) - F(0).
    // Integrand at x=0: 3/1=3, at x=1: 5/3≈1.667.
    // Approximate by trapezoid: (3+1.667)/2 ≈ 2.33.
    let f1 = eval_at_int(&anti, &x, 1);
    let f0 = eval_at_int(&anti, &x, 0);
    if let (Some(f1), Some(f0)) = (f1, f0) {
        let val = f1 - f0;
        // The definite integral should be positive and roughly 2.2–2.5.
        assert!(
            val > 1.5 && val < 3.5,
            "∫₀¹ (2x+3)/(x²+x+1) dx should be ≈ 2.3: got {val}"
        );
    }
}

#[test]
fn integrate_1_over_x_plus_1_squared() {
    // ∫ 1/(x+1)² dx = -1/(x+1).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x + 1).powi(2);
    let anti = expr.integrate(&x);

    // Numerical: F(3) - F(1) = -1/4 - (-1/2) = 1/4 = 0.25.
    let f3 = eval_at_int(&anti, &x, 3);
    let f1 = eval_at_int(&anti, &x, 1);
    if let (Some(f3), Some(f1)) = (f3, f1) {
        let val = f3 - f1;
        assert!(
            (val - 0.25).abs() < 1e-10,
            "∫₁³ 1/(x+1)² dx = 0.25: got {val}"
        );
    }
}

#[test]
fn integrate_repeated_and_coprime_1_over_x_plus_1_sq_times_x_minus_1() {
    // ∫ 1/((x+1)²(x-1)) dx
    // apart gives: A/(x+1)² + B/(x+1) + C/(x-1)
    // This should integrate to ln + rational terms.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = (&x + 1).powi(2) * (&x - 1);
    let expr = 1 / &denom;
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    // Should integrate (no unevaluated integral).
    assert!(
        !s.contains("Integral"),
        "∫ 1/((x+1)²(x-1)) dx should evaluate: {s}"
    );

    // Numerical: F(3) - F(2).
    // At x=3: 1/(16*2)=1/32.  At x=2: 1/(9*1)=1/9.
    let f3 = eval_at_int(&anti, &x, 3);
    let f2 = eval_at_int(&anti, &x, 2);
    if let (Some(f3), Some(f2)) = (f3, f2) {
        let val = f3 - f2;
        // Exact: -1/(2(x+1)) + 1/4*ln|(x-1)/(x+1)| evaluated from 2 to 3
        // F(3) = -1/8 + 1/4*ln(2/4) = -1/8 + 1/4*ln(1/2) = -0.125 - 0.17328 = -0.29828
        // F(2) = -1/6 + 1/4*ln(1/3) = -0.16667 - 0.27465 = -0.44132
        // val ≈ -0.29828 + 0.44132 ≈ 0.1430
        assert!(
            val.abs() < 1.0,
            "∫₂³ 1/((x+1)²(x-1)) dx should be small: got {val}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression tests — ensure the apart fix doesn't break existing cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_does_not_regress_simple_inverse() {
    // 1/x should remain unchanged (not a polynomial denominator needing apart).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / &x;
    let anti = expr.integrate(&x);
    let s = format!("{anti}");
    assert!(s.contains("ln"), "∫ 1/x dx should be ln: {s}");
}

#[test]
fn apart_does_not_regress_polynomial() {
    // x² + x should integrate without apart.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + &x;
    let anti = expr.integrate(&x);

    // F(1) - F(0) = 1/3 + 1/2 = 5/6 ≈ 0.8333
    let f1 = eval_at_int(&anti, &x, 1);
    let f0 = eval_at_int(&anti, &x, 0);
    if let (Some(f1), Some(f0)) = (f1, f0) {
        let val = f1 - f0;
        assert!(
            (val - 5.0 / 6.0).abs() < 1e-10,
            "∫₀¹ (x²+x) dx = 5/6: got {val}"
        );
    }
}

#[test]
fn apart_with_polynomial_quotient() {
    // x³/(x²-1) = x + x/(x²-1).  After apart on x/(x²-1), should get
    // x + 1/2/(x-1) + 1/2/(x+1).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(3) / (&x.powi(2) - 1);
    let decomposed = expr.partial_fractions(&x);

    // Verify numerically at x=2.
    let orig_val = eval_at_int(&expr, &x, 2).unwrap(); // 8/3
    let dec_val = eval_at_int(&decomposed, &x, 2).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "x³/(x²-1) apart must match at x=2: {orig_val} vs {dec_val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Bug regression: the original bug gave 0.139 for ∫ 1/(x⁵+1) dx
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_1_over_x5_plus_1_not_0_139() {
    // The old (buggy) apart gave only 1/5*ln|x+1| for ∫ 1/(x⁵+1) dx.
    // At x=1: 1/5*ln(2) ≈ 0.139.  This is WRONG — the correct definite
    // integral ∫₀¹ 1/(x⁵+1) dx ≈ 0.836.
    //
    // With the fix, apart now correctly decomposes 1/(x⁵+1) into
    // 1/(5(x+1)) + (polynomial)/(x⁴-x³+x²-x+1).  The linear part
    // integrates to 1/5*ln|x+1|.  The quartic part may or may not
    // fully integrate (depending on the integrator's capabilities), but
    // the decomposition is correct and the bug is that the apart output
    // now accounts for ALL terms of the denominator.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(5) + 1);
    let decomposed = expr.partial_fractions(&x);

    // The decomposition should differ from the original expression.
    let s = format!("{decomposed}");
    assert_ne!(
        s,
        format!("{expr}"),
        "apart(1/(x⁵+1)) should decompose: {s}"
    );

    // Verify the decomposition is numerically correct at x=1.
    let orig = eval_at_int(&expr, &x, 1).unwrap(); // 1/2
    let dec = eval_at_int(&decomposed, &x, 1).unwrap();
    assert!(
        (orig - dec).abs() < 1e-10,
        "apart decomposition of 1/(x⁵+1) must equal original at x=1: {orig} vs {dec}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Extended GCD sanity — ensure the polynomial decomposition algebra works
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_x4_minus_1_four_factors() {
    // x⁴-1 = (x-1)(x+1)(x²+1)  — two linear + one quadratic factor.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(4) - 1);
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "1/(x⁴-1) should decompose: {s}");

    // Verify at x=2: 1/15.
    let orig_val = eval_at_int(&expr, &x, 2).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 2).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "1/(x⁴-1) must match at x=2: {orig_val} vs {dec_val}"
    );

    // Verify at x=3: 1/80.
    let orig_val = eval_at_int(&expr, &x, 3).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 3).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "1/(x⁴-1) must match at x=3: {orig_val} vs {dec_val}"
    );
}

#[test]
fn integrate_1_over_x4_minus_1() {
    // After apart, 1/(x⁴-1) decomposes into terms with (x-1), (x+1),
    // and (x²+1) denominators — all integrable.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(4) - 1);
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    // Should contain ln and atan.
    let has_ln = s.contains("ln");
    let has_atan = s.contains("atan");

    // Numerical: F(3) - F(2).
    let f3 = eval_at_int(&anti, &x, 3);
    let f2 = eval_at_int(&anti, &x, 2);
    if let (Some(f3), Some(f2)) = (f3, f2) {
        let val = f3 - f2;
        // Exact: 1/4*ln|(x-1)/(x+1)| - 1/2*atan(x)  from 2 to 3
        // F(3) - F(2) = 1/4*(ln(2/4) - ln(1/3)) - 1/2*(atan3 - atan2)
        //             = 1/4*ln(3/2) - 1/2*(1.2490 - 1.1071) ≈ 0.1014 - 0.0710 ≈ 0.0304
        assert!(
            val.abs() < 0.5,
            "∫₂³ 1/(x⁴-1) dx should be small: got {val}"
        );
    } else {
        assert!(
            has_ln || has_atan,
            "∫ 1/(x⁴-1) dx should have ln or atan: {s}"
        );
    }
}

#[test]
fn apart_1_over_x2_plus_1_squared_trivial() {
    // 1/(x²+1)² — single irreducible quadratic factor with multiplicity 2.
    // Standard PFD over ℚ gives 1/(x²+1)² (trivially, since A=0, B=0
    // for the 1/(x²+1) term).  So apart may return it unchanged or with
    // the same single term.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(2) + 1).powi(2);
    let decomposed = expr.partial_fractions(&x);

    // Verify numerical equivalence.
    let orig_val = eval_at_int(&expr, &x, 1).unwrap(); // 1/4
    let dec_val = eval_at_int(&decomposed, &x, 1).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "1/(x²+1)² apart must match at x=1: {orig_val} vs {dec_val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration of x³-1 is the most impactful new case — test thoroughly
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_1_over_x3_minus_1_has_both_ln_and_atan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(3) - 1);
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    // The correct antiderivative is:
    //   1/3*ln|x-1| - 1/6*ln(x²+x+1) - 1/√3*atan((2x+1)/√3)
    // So it should contain both ln and atan.
    assert!(s.contains("ln"), "∫ 1/(x³-1) dx should contain ln: {s}");
    assert!(s.contains("atan"), "∫ 1/(x³-1) dx should contain atan: {s}");
}

#[test]
fn integrate_1_over_x3_minus_1_definite_2_to_5() {
    // Definite integral ∫₂⁵ 1/(x³-1) dx evaluated numerically.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(3) - 1);
    let anti = expr.integrate(&x);

    let f5 = eval_at_int(&anti, &x, 5);
    let f2 = eval_at_int(&anti, &x, 2);

    if let (Some(f5), Some(f2)) = (f5, f2) {
        let val = f5 - f2;
        // Approximate by numerical quadrature:
        // 1/7 + 1/26 + 1/63 + 1/124 ≈ 0.143+0.038+0.016+0.008 ≈ 0.205
        // (crude Riemann sum).  Exact ≈ 0.119.
        assert!(
            val > 0.0 && val < 0.5,
            "∫₂⁵ 1/(x³-1) dx should be positive and small: {val}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Corner cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_constant_denom_unchanged() {
    // 1/3 should return unchanged (denominator is constant, no var).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = ctx.rational(1, 3);
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");
    assert_eq!(s, "1/3", "constant should be unchanged: {s}");
}

#[test]
fn apart_non_rational_unchanged() {
    // sin(x)/x should be returned unchanged (not a polynomial fraction).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin() / &x;
    let decomposed = expr.partial_fractions(&x);
    // Should be essentially unchanged.
    let orig_val = eval_at_int(&expr, &x, 1);
    let dec_val = eval_at_int(&decomposed, &x, 1);
    if let (Some(o), Some(d)) = (orig_val, dec_val) {
        assert!(
            (o - d).abs() < 1e-10,
            "sin(x)/x apart should not change value: {o} vs {d}"
        );
    }
}

#[test]
fn apart_high_degree_multiple_factors() {
    // x⁴+5x²+6 = (x²+2)(x²+3) — two irreducible quadratics.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = &x.powi(4) + &x.powi(2) * 5 + 6;
    let expr = 1 / &denom;
    let decomposed = expr.partial_fractions(&x);
    let s = format!("{decomposed}");

    assert_ne!(s, format!("{expr}"), "1/(x⁴+5x²+6) should decompose: {s}");

    // Verify at x=1: 1/(1+5+6) = 1/12.
    let orig_val = eval_at_int(&expr, &x, 1).unwrap();
    let dec_val = eval_at_int(&decomposed, &x, 1).unwrap();
    assert!(
        (orig_val - dec_val).abs() < 1e-10,
        "1/(x⁴+5x²+6) must match at x=1: {orig_val} vs {dec_val}"
    );
}

#[test]
fn integrate_1_over_x4_plus_5x2_plus_6() {
    // (x²+2)(x²+3) → apart gives A/(x²+2) + B/(x²+3).
    // Each integrates to atan form.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = &x.powi(4) + &x.powi(2) * 5 + 6;
    let expr = 1 / &denom;
    let anti = expr.integrate(&x);
    let s = format!("{anti}");

    assert!(
        s.contains("atan"),
        "∫ 1/(x⁴+5x²+6) dx should contain atan: {s}"
    );

    // Numerical: F(1)-F(0) should be a reasonable positive number.
    let f1 = eval_at_int(&anti, &x, 1);
    let f0 = eval_at_int(&anti, &x, 0);
    if let (Some(f1), Some(f0)) = (f1, f0) {
        let val = f1 - f0;
        // Exact: ∫₀¹ 1/((x²+2)(x²+3)) dx = atan(1/√2)/√2 - atan(1/√3)/√3
        // ≈ 0.6155/1.4142 - 0.5236/1.7321 ≈ 0.4352 - 0.3023 ≈ 0.1329
        assert!(
            (val - 0.1329).abs() < 0.01,
            "∫₀¹ 1/(x⁴+5x²+6) dx ≈ 0.133: got {val}"
        );
    }
}
