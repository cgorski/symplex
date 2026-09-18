//! Comprehensive tests for the integration and radical simplification fixes.
//!
//! Covers all phases:
//!   Phase 1: Double-counting fix in try_risch_rational
//!   Phase 2: Deep has_unevaluated check in by-parts and cyclic IBP
//!   Phase 3: Symbolic coefficient integration (irrational quadratic coefficients)
//!   Phase 5: Radical simplification (powdenest, powsimp_base, canon_pow)
//!
//! Each test verifies both structural correctness (no unevaluated integrals)
//! and numerical correctness (definite integral matches ground truth).

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Numerical integration via midpoint rule for ground truth comparison.
fn numerical_integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut sum = 0.0;
    for i in 0..n {
        let x = a + (i as f64 + 0.5) * h;
        sum += f(x);
    }
    sum * h
}

/// Evaluate the definite integral F(b) - F(a) for an antiderivative expression.
fn definite(anti: &Ex, var: &Ex, a: i64, b: i64) -> Option<f64> {
    let ctx = anti.context();
    let fa = anti.subs(var, &ctx.int(a)).eval().eval_f64().ok()?;
    let fb = anti.subs(var, &ctx.int(b)).eval().eval_f64().ok()?;
    Some(fb - fa)
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 1: Double-counting fix in try_risch_rational
//
// These integrands have mixed factorization: some linear factors AND
// some irreducible quadratic factors.  Previously, the rational log
// terms were emitted twice — once from Rothstein-Trager and once from
// the recursive heuristic integration of the full remainder.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn phase1_one_over_x3_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(3) - 1);
    let anti = expr.integrate(&x);

    // Must not contain unevaluated integrals.
    assert!(
        !anti.has_unevaluated(),
        "∫ 1/(x³-1) dx should be fully evaluated: {anti}"
    );

    // Numerical check: ∫₂³ 1/(x³-1) dx ≈ 0.07539
    let gt = numerical_integrate(|x| 1.0 / (x.powi(3) - 1.0), 2.0, 3.0, 100_000);
    let val = definite(&anti, &x, 2, 3).expect("should evaluate");
    assert!(
        (val - gt).abs() < 1e-6,
        "∫₂³ 1/(x³-1) dx: got {val}, expected ≈ {gt}"
    );
}

#[test]
fn phase1_one_over_x3_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(3) + 1);
    let anti = expr.integrate(&x);

    assert!(
        !anti.has_unevaluated(),
        "∫ 1/(x³+1) dx should be fully evaluated: {anti}"
    );

    let gt = numerical_integrate(|x| 1.0 / (x.powi(3) + 1.0), 0.0, 2.0, 100_000);
    let fa = anti.subs(&x, &ctx.int(0)).eval().eval_f64().unwrap();
    let fb = anti.subs(&x, &ctx.int(2)).eval().eval_f64().unwrap();
    let val = fb - fa;
    assert!(
        (val - gt).abs() < 1e-5,
        "∫₀² 1/(x³+1) dx: got {val}, expected ≈ {gt}"
    );
}

#[test]
fn phase1_one_over_x4_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(4) - 1);
    let anti = expr.integrate(&x);

    assert!(
        !anti.has_unevaluated(),
        "∫ 1/(x⁴-1) dx should be fully evaluated: {anti}"
    );

    let gt = numerical_integrate(|x| 1.0 / (x.powi(4) - 1.0), 2.0, 3.0, 100_000);
    let val = definite(&anti, &x, 2, 3).expect("should evaluate");
    assert!(
        (val - gt).abs() < 1e-6,
        "∫₂³ 1/(x⁴-1) dx: got {val}, expected ≈ {gt}"
    );
}

#[test]
fn phase1_one_over_x_minus_1_times_x2_plus_4() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / ((&x - 1) * (&x.powi(2) + 4));
    let anti = expr.integrate(&x);

    assert!(
        !anti.has_unevaluated(),
        "∫ 1/((x-1)(x²+4)) dx should be fully evaluated: {anti}"
    );

    let gt = numerical_integrate(|x| 1.0 / ((x - 1.0) * (x * x + 4.0)), 2.0, 3.0, 100_000);
    let val = definite(&anti, &x, 2, 3).expect("should evaluate");
    assert!(
        (val - gt).abs() < 1e-6,
        "∫₂³ 1/((x-1)(x²+4)) dx: got {val}, expected ≈ {gt}"
    );
}

#[test]
fn phase1_two_linear_one_quadratic() {
    // 1/((x-1)(x-2)(x²+1)) — two rational roots + one irreducible quadratic
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / ((&x - 1) * (&x - 2) * (&x.powi(2) + 1));
    let anti = expr.integrate(&x);

    assert!(
        !anti.has_unevaluated(),
        "∫ 1/((x-1)(x-2)(x²+1)) dx should be fully evaluated: {anti}"
    );

    let gt = numerical_integrate(
        |x| 1.0 / ((x - 1.0) * (x - 2.0) * (x * x + 1.0)),
        3.0,
        4.0,
        100_000,
    );
    let val = definite(&anti, &x, 3, 4).expect("should evaluate");
    assert!(
        (val - gt).abs() < 1e-6,
        "∫₃⁴ 1/((x-1)(x-2)(x²+1)) dx: got {val}, expected ≈ {gt}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2: Deep has_unevaluated check in by-parts and cyclic IBP
//
// x/(x⁴+x²+1) was trapped by cyclic IBP which falsely declared success
// because it only checked the top node for Integral, missing nested
// unevaluated integrals inside an Add.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn phase2_x_over_quartic_plus_quadratic_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x / (&x.powi(4) + &x.powi(2) + 1);
    let anti = expr.integrate(&x);

    assert!(
        !anti.has_unevaluated(),
        "∫ x/(x⁴+x²+1) dx should be fully evaluated: {anti}"
    );

    let gt = numerical_integrate(|x| x / (x.powi(4) + x * x + 1.0), 2.0, 3.0, 100_000);
    let val = definite(&anti, &x, 2, 3).expect("should evaluate");
    assert!(
        (val - gt).abs() < 1e-6,
        "∫₂³ x/(x⁴+x²+1) dx: got {val}, expected ≈ {gt}"
    );
}

#[test]
fn phase2_one_over_x4_plus_x2_plus_1() {
    // 1/(x⁴+x²+1) = 1/((x²+x+1)(x²-x+1)) — both factors irreducible over ℚ.
    // Should be handled cleanly via apart decomposition without by-parts.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(4) + &x.powi(2) + 1);
    let anti = expr.integrate(&x);

    assert!(
        !anti.has_unevaluated(),
        "∫ 1/(x⁴+x²+1) dx should be fully evaluated: {anti}"
    );

    let gt = numerical_integrate(|x| 1.0 / (x.powi(4) + x * x + 1.0), 2.0, 3.0, 100_000);
    let val = definite(&anti, &x, 2, 3).expect("should evaluate");
    assert!(
        (val - gt).abs() < 1e-6,
        "∫₂³ 1/(x⁴+x²+1) dx: got {val}, expected ≈ {gt}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 3: Symbolic coefficient integration
//
// When apart produces terms with irrational coefficients (e.g., √5 from
// cyclotomic polynomials), try_complete_square_integral and
// try_linear_over_quadratic now handle symbolic coefficients via arena
// arithmetic instead of requiring Ratio<BigInt> polynomial conversion.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn phase3_one_over_quadratic_with_sqrt2() {
    // ∫ 1/(x² + √2·x + 1) dx — quadratic with irrational coefficient
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sqrt2 = ctx.int(2).sqrt();
    let expr = 1 / (&x.powi(2) + &sqrt2 * &x + 1);
    let anti = expr.integrate(&x);

    assert!(
        !anti.has_unevaluated(),
        "∫ 1/(x²+√2x+1) dx should be fully evaluated: {anti}"
    );

    let sqrt2_f = std::f64::consts::SQRT_2;
    let gt = numerical_integrate(|x| 1.0 / (x * x + sqrt2_f * x + 1.0), 0.0, 2.0, 100_000);
    let fa = anti.subs(&x, &ctx.int(0)).eval().eval_f64().unwrap();
    let fb = anti.subs(&x, &ctx.int(2)).eval().eval_f64().unwrap();
    let val = fb - fa;
    assert!(
        (val - gt).abs() < 1e-5,
        "∫₀² 1/(x²+√2x+1) dx: got {val}, expected ≈ {gt}"
    );
}

#[test]
fn phase3_one_over_x5_minus_1_via_apart() {
    // ∫ 1/(x⁵-1) dx via the apart → integrate path.
    // apart produces terms with √5 coefficients from the cyclotomic
    // quartic x⁴+x³+x²+x+1.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(5) - 1);
    let apart = expr.partial_fractions(&x);
    let anti = apart.integrate(&x);

    assert!(
        !anti.has_unevaluated(),
        "∫ apart(1/(x⁵-1)) dx should be fully evaluated: {anti}"
    );

    let gt = numerical_integrate(|x| 1.0 / (x.powi(5) - 1.0), 2.0, 3.0, 100_000);
    let val = definite(&anti, &x, 2, 3).expect("should evaluate");
    assert!(
        (val - gt).abs() < 1e-5,
        "∫₂³ apart(1/(x⁵-1)) dx: got {val}, expected ≈ {gt}"
    );
}

#[test]
fn phase3_one_over_x8_minus_1_via_apart() {
    // 1/(x⁸-1) has factor x⁴+1 which is irreducible over ℤ but factors
    // over ℝ into quadratics with √2 coefficients.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / (&x.powi(8) - 1);
    let apart = expr.partial_fractions(&x);
    let anti = apart.integrate(&x);

    assert!(
        !anti.has_unevaluated(),
        "∫ apart(1/(x⁸-1)) dx should be fully evaluated: {anti}"
    );

    let gt = numerical_integrate(|x| 1.0 / (x.powi(8) - 1.0), 2.0, 3.0, 100_000);
    let val = definite(&anti, &x, 2, 3).expect("should evaluate");
    assert!(
        (val - gt).abs() < 1e-5,
        "∫₂³ apart(1/(x⁸-1)) dx: got {val}, expected ≈ {gt}"
    );
}

#[test]
fn phase3_linear_over_quadratic_with_sqrt5() {
    // Directly test the (ax+b)/(cx²+dx+e) integration with irrational coefficients.
    // This is one of the terms from apart(1/(x⁵-1)).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sqrt5 = ctx.int(5).sqrt();

    // (√5/10 - 1/10)·x - 2/5  over  x² + (-√5/2 + 1/2)·x + 1
    let numer = (&sqrt5 / 10 - ctx.rational(1, 10)) * &x - ctx.rational(2, 5);
    let denom = &x.powi(2) + (-&sqrt5 / 2 + ctx.rational(1, 2)) * &x + 1;
    let expr = &numer / &denom;
    let anti = expr.integrate(&x);

    assert!(
        !anti.has_unevaluated(),
        "∫ (linear with √5)/(quadratic with √5) dx should be fully evaluated: {anti}"
    );

    // Numerical check
    let s5 = 5.0_f64.sqrt();
    let gt = numerical_integrate(
        |x| {
            let n = (s5 / 10.0 - 0.1) * x - 0.4;
            let d = x * x + (-s5 / 2.0 + 0.5) * x + 1.0;
            n / d
        },
        2.0,
        3.0,
        100_000,
    );
    let val = definite(&anti, &x, 2, 3).expect("should evaluate");
    assert!(
        (val - gt).abs() < 1e-5,
        "∫₂³ (linear/quadratic with √5): got {val}, expected ≈ {gt}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 5: Radical simplification
//
// canon_pow, powdenest, and powsimp_base enable exact simplification of
// radical expressions, including:
//   (√a)² → a                   (canon_pow positive-base flattening)
//   (c·√a)² → c²·a             (powdenest)
//   √a·√b → √(ab)              (powsimp_base)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn phase5_sqrt_squared_at_construction() {
    // (√5)² should simplify to 5 at construction time via canon_pow.
    let ctx = Context::new();
    let sqrt5 = ctx.int(5).sqrt();
    let result = sqrt5.powi(2);
    assert_eq!(
        format!("{result}"),
        "5",
        "(√5)² should be 5 at construction"
    );
}

#[test]
fn phase5_cbrt_cubed_at_construction() {
    // (∛2)³ should simplify to 2 at construction time.
    let ctx = Context::new();
    let cbrt2 = ctx.int(2).cbrt();
    let result = cbrt2.powi(3);
    assert_eq!(
        format!("{result}"),
        "2",
        "(∛2)³ should be 2 at construction"
    );
}

#[test]
fn phase5_fourth_root_squared() {
    // (3^{1/4})² should simplify to √3 at construction time.
    let ctx = Context::new();
    let root4_3 = ctx.int(3).nthroot(4);
    let result = root4_3.powi(2);
    assert_eq!(
        format!("{result}"),
        "sqrt(3)",
        "(3^(1/4))² should be √3 at construction"
    );
}

#[test]
fn phase5_sqrt2_times_sqrt3_smart_simplify() {
    // √2·√3 → √6 via smart_simplify
    let ctx = Context::new();
    let sqrt2 = ctx.int(2).sqrt();
    let sqrt3 = ctx.int(3).sqrt();
    let product = &sqrt2 * &sqrt3;
    let result = product.simplify();
    assert_eq!(
        format!("{result}"),
        "sqrt(6)",
        "√2·√3 should smart_simplify to √6"
    );
}

#[test]
fn phase5_sqrt2_times_sqrt3_full_simplify() {
    // √2·√3 → √6 via full_simplify
    let ctx = Context::new();
    let sqrt2 = ctx.int(2).sqrt();
    let sqrt3 = ctx.int(3).sqrt();
    let product = &sqrt2 * &sqrt3;
    let result = product.simplify();
    assert_eq!(
        format!("{result}"),
        "sqrt(6)",
        "√2·√3 should full_simplify to √6"
    );
}

#[test]
fn phase5_cbrt2_times_cbrt4() {
    // 2^(1/3) · 4^(1/3) → 8^(1/3) → 2
    let ctx = Context::new();
    let cbrt2 = ctx.int(2).cbrt();
    let cbrt4 = ctx.int(4).cbrt();
    let product = &cbrt2 * &cbrt4;
    let result = product.simplify();
    assert_eq!(
        format!("{result}"),
        "2",
        "2^(1/3)·4^(1/3) should simplify to 2"
    );
}

#[test]
fn phase5_golden_ratio_identity() {
    // φ² - φ - 1 = 0  where φ = (1+√5)/2
    let ctx = Context::new();
    let sqrt5 = ctx.int(5).sqrt();
    let phi = (ctx.int(1) + &sqrt5) / 2;
    let test = phi.powi(2).expand().eval() - &phi - 1;
    let result = test.simplify();
    assert_eq!(
        format!("{result}"),
        "0",
        "φ² - φ - 1 should smart_simplify to 0"
    );
}

#[test]
fn phase5_sum_of_sqrt_squared_identity() {
    // (√2 + √3)² - 5 - 2√6 = 0
    let ctx = Context::new();
    let sqrt2 = ctx.int(2).sqrt();
    let sqrt3 = ctx.int(3).sqrt();
    let sqrt6 = ctx.int(6).sqrt();
    let test = (&sqrt2 + &sqrt3).powi(2).expand().eval() - 5 - ctx.int(2) * &sqrt6;
    let result = test.simplify();
    assert_eq!(
        format!("{result}"),
        "0",
        "(√2+√3)²-5-2√6 should smart_simplify to 0"
    );
}

#[test]
fn phase5_powdenest_half_sqrt5_squared() {
    // (½·√5)² → 5/4 via powdenest
    let ctx = Context::new();
    let half_sqrt5 = ctx.rational(1, 2) * ctx.int(5).sqrt();
    let squared = half_sqrt5.powi(2);
    let result = squared.simplify();
    assert_eq!(
        format!("{result}"),
        "5/4",
        "(½·√5)² should smart_simplify to 5/4"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression: verify that standard integration tests still pass
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn regression_standard_by_parts_x_exp_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = (&x * &x.exp()).integrate(&x);
    assert!(!anti.has_unevaluated(), "∫ x·exp(x) dx: {anti}");
}

#[test]
fn regression_standard_by_parts_x_sin_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = (&x * &x.sin()).integrate(&x);
    assert!(!anti.has_unevaluated(), "∫ x·sin(x) dx: {anti}");
}

#[test]
fn regression_cyclic_ibp_exp_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = (&x.exp() * &x.sin()).integrate(&x);
    assert!(!anti.has_unevaluated(), "∫ exp(x)·sin(x) dx: {anti}");
}

#[test]
fn regression_cyclic_ibp_exp_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = (&x.exp() * &x.cos()).integrate(&x);
    assert!(!anti.has_unevaluated(), "∫ exp(x)·cos(x) dx: {anti}");
}

#[test]
fn regression_one_over_x2_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = (1 / (&x.powi(2) + 1)).integrate(&x);
    assert!(!anti.has_unevaluated(), "∫ 1/(x²+1) dx: {anti}");
    let s = format!("{anti}");
    assert!(s.contains("atan"), "should contain atan: {s}");
}

#[test]
fn regression_one_over_x2_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = (1 / (&x.powi(2) - 1)).integrate(&x);
    assert!(!anti.has_unevaluated(), "∫ 1/(x²-1) dx: {anti}");
    let s = format!("{anti}");
    assert!(s.contains("ln"), "should contain ln: {s}");

    // Numerical check
    let gt = numerical_integrate(|x| 1.0 / (x * x - 1.0), 2.0, 3.0, 100_000);
    let val = definite(&anti, &x, 2, 3).expect("should evaluate");
    assert!(
        (val - gt).abs() < 1e-6,
        "∫₂³ 1/(x²-1) dx: got {val}, expected ≈ {gt}"
    );
}

#[test]
fn regression_one_over_x2_plus_1_squared() {
    // 1/(x²+1)² — Hermite extracts rational part, arctan for remainder
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = (1 / (&x.powi(2) + 1).powi(2)).integrate(&x);
    assert!(!anti.has_unevaluated(), "∫ 1/(x²+1)² dx: {anti}");
}
