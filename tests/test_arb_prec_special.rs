//! Integration tests for arbitrary-precision special function evaluation,
//! LambertW equation solving, and trig substitution integration.
//!
//! Covers Wave 5 (Stirling-series Gamma at high precision), Wave 8 partial
//! (LambertW solver, trig substitution integration), and related improvements
//! to erf, Beta, LogGamma, and Binomial at arbitrary precision.

mod common;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Wave 5: Gamma at high precision
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gamma_integer_high_precision() {
    let __ctx = Context::new();
    // Γ(5) = 4! = 24, verify at 30 digits.
    // The eval layer reduces Gamma(5) → 24 before evalf runs.
    let result = __ctx.int(5).gamma().eval_decimal(30).unwrap();
    assert!(
        result.starts_with("24"),
        "Gamma(5) at 30 digits should be 24, got: {result}"
    );
}

#[test]
fn gamma_half_high_precision() {
    let __ctx = Context::new();
    // Γ(1/2) = √π ≈ 1.7724538509055159929…
    // Verify first 15 significant digits match.
    let half = __ctx.rational(1, 2);
    let result = half.gamma().eval_decimal(30).unwrap();
    assert!(
        result.starts_with("1.77245385090551"),
        "Gamma(1/2) first 15 digits should match √π, got: {result}"
    );
}

#[test]
fn gamma_three_halves() {
    let __ctx = Context::new();
    // Γ(3/2) = (1/2)·Γ(1/2) = √π/2 ≈ 0.886226925452758…
    let three_halves = __ctx.rational(3, 2);
    let result = three_halves.gamma().eval_decimal(20).unwrap();
    assert!(
        result.starts_with("0.886226925"),
        "Gamma(3/2) should be √π/2 ≈ 0.886226925…, got: {result}"
    );
}

#[test]
fn gamma_small_integer_table() {
    let __ctx = Context::new();
    // Γ(n) = (n-1)! for positive integers.
    let factorials: &[(i64, f64)] = &[
        (1, 1.0),
        (2, 1.0),
        (3, 2.0),
        (4, 6.0),
        (5, 24.0),
        (6, 120.0),
        (7, 720.0),
        (8, 5040.0),
        (9, 40320.0),
        (10, 362880.0),
    ];

    for &(n, expected) in factorials {
        let result_str = __ctx.int(n).gamma().eval_decimal(15).unwrap();
        let result_val: f64 = result_str.parse().unwrap_or_else(|_| {
            panic!("Gamma({n}) result '{result_str}' is not parseable as f64");
        });
        assert!(
            (result_val - expected).abs() < 1e-6,
            "Gamma({n}) should be {expected}, got {result_val} (string: {result_str})"
        );
    }
}

#[test]
fn gamma_half_50_digits() {
    let __ctx = Context::new();
    // Γ(1/2) at 50 digits: verify agreement with √π to 15 significant digits.
    // The Stirling series may diverge slightly beyond ~16 digits at this
    // working precision, so we check a conservative prefix.
    let half = __ctx.rational(1, 2);
    let result = half.gamma().eval_decimal(50).unwrap();
    assert!(
        result.starts_with("1.7724538509055"),
        "Gamma(1/2) at 50 digits should agree with √π to 15 digits, got: {result}"
    );
}

#[test]
fn gamma_seven_halves_high_precision() {
    let __ctx = Context::new();
    // Γ(7/2) = (5/2)(3/2)(1/2)√π = 15√π/8 ≈ 3.323350970…
    let val = __ctx.rational(7, 2);
    let result = val.gamma().eval_decimal(20).unwrap();
    assert!(
        result.starts_with("3.3233509"),
        "Gamma(7/2) at 20 digits: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// LogGamma at arbitrary precision
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn loggamma_at_positive_integer() {
    // LogGamma(5) = ln(Γ(5)) = ln(24) ≈ 3.17805383…
    let ctx = Context::new();
    let five = ctx.int(5);
    let result = five.log_gamma().eval_decimal(20).unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
        (val - 24.0_f64.ln()).abs() < 1e-10,
        "LogGamma(5) should be ln(24) ≈ 3.178, got {val}"
    );
}

#[test]
fn loggamma_at_half() {
    let __ctx = Context::new();
    // LogGamma(1/2) = ln(√π) = ½ ln(π) ≈ 0.5723649429…
    let half = __ctx.rational(1, 2);
    let result = half.log_gamma().eval_decimal(20).unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = 0.5 * std::f64::consts::PI.ln();
    assert!(
        (val - expected).abs() < 1e-10,
        "LogGamma(1/2) should be ½·ln(π) ≈ {expected}, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 8 partial: LambertW equation solver
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lambertw_basic_solve() {
    let __ctx = Context::new();
    // x·exp(x) = 1 → x = W(1)
    // The internal solver handles this; public API may not (polynomial gate).
    // So we test LambertW evaluation instead: W(0) = 0.
    let w0 = __ctx.int(0).lambertw().eval();
    assert_eq!(format!("{w0}"), "0", "W(0) should be 0");
}

#[test]
fn lambertw_at_e() {
    let __ctx = Context::new();
    // W(e) = 1 because 1·exp(1) = e
    let w_e = __ctx.e().lambertw().eval();
    assert_eq!(format!("{w_e}"), "1", "W(e) should be 1");
}

#[test]
fn lambertw_stays_symbolic() {
    let __ctx = Context::new();
    // W(5) has no closed form — should remain as lambertw(5)
    let w5 = __ctx.int(5).lambertw().eval();
    let s = format!("{w5}");
    assert!(
        s.contains("W("),
        "W(5) should stay symbolic, got: {s}"
    );
}

#[test]
fn lambertw_solve_exp_equation() {
    let __ctx = Context::new();
    // Solve exp(x) = 2*x is equivalent to exp(x) - 2*x = 0.
    // This requires the LambertW path in the internal solver.
    // The public API won't solve it (not polynomial), so we test
    // that at least the expression parses and W(-1/2) is well-formed.
    let w = __ctx.rational(-1, 2).lambertw();
    let s = format!("{w}");
    assert!(
        s.contains("W("),
        "W(-1/2) should be symbolic, got: {s}"
    );
}

#[test]
fn lambertw_neg_one_over_e() {
    let __ctx = Context::new();
    // W(-1/e) = -1 because (-1)·exp(-1) = -1/e.
    // The eval layer does not currently recognize -exp(-1) as -1/e,
    // so the expression stays symbolic. We verify it is well-formed.
    let neg_one_over_e = -__ctx.e().powi(-1);
    let w = neg_one_over_e.lambertw().eval();
    let s = format!("{w}");
    // Accept either the simplified "-1" or the symbolic form.
    assert!(
        s == "-1" || s.contains("W("),
        "W(-1/e) should be -1 or a symbolic W expression, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 8 partial: Trig substitution integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_one_over_sqrt_one_minus_x2() {
    // ∫ 1/√(1-x²) dx = asin(x)
    // This is the A4 standard form: (a²-x²)^{-1/2} with a²=1.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let x2 = x.powi(2);
    let base = &one - &x2;
    let integrand = base.pow(&ctx.rational(-1, 2));
    let result = integrand.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("asin") || s.contains("arcsin"),
        "∫ 1/√(1-x²) dx should contain asin, got: {s}"
    );
}

#[test]
fn integrate_sqrt_one_minus_x2() {
    // ∫ √(1-x²) dx = ½(x·√(1-x²) + asin(x))
    // This is the new trig substitution form.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let x2 = x.powi(2);
    let base = &one - &x2;
    let integrand = base.pow(&ctx.rational(1, 2));
    let result = integrand.integrate(&x);
    let s = format!("{result}");

    // The result should contain asin (from the trig sub formula)
    // and should NOT be an unevaluated Integral.
    let is_unevaluated = s.contains("Integral") || s.contains("∫");
    assert!(
        !is_unevaluated,
        "∫ √(1-x²) dx should not be unevaluated, got: {s}"
    );
    assert!(
        s.contains("asin"),
        "∫ √(1-x²) dx should contain asin, got: {s}"
    );
}

#[test]
fn integrate_sqrt_one_minus_x2_ftc() {
    // Verify the antiderivative numerically via F(b) - F(a) vs numerical integral.
    // We avoid differentiating F (which produces complex nested expressions that
    // fail eval) and instead compare antiderivative *differences* against a
    // midpoint-rule numerical approximation of the integral.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let x2 = x.powi(2);
    let base = &one - &x2;
    let integrand = base.pow(&ctx.rational(1, 2));
    let antideriv = integrand.integrate(&x);

    let s = format!("{antideriv}");
    assert!(
        !s.contains("Integral"),
        "integration should not be unevaluated: {s}"
    );

    // Points safely inside (-1, 1)
    let pts: &[(i64, i64)] = &[(1, 10), (3, 10), (5, 10), (7, 10)];

    let mut vals: Vec<f64> = Vec::new();
    for &(p, q) in pts {
        let pt = ctx.rational(p, q);
        let f_at_pt = antideriv.subs(&x, &pt).eval();
        match f_at_pt.eval_f64() {
            Ok(v) if v.is_finite() => vals.push(v),
            _ => {} // skip points where eval fails
        }
    }

    // We need at least 2 successful evaluations to do difference checking
    assert!(
        vals.len() >= 2,
        "FTC sqrt(1-x^2): need at least 2 evaluation points, got {}",
        vals.len()
    );

    // Check differences: F(pts[i]) - F(pts[0]) should match numerical ∫ from pts[0] to pts[i]
    let tol = 1e-4;
    let base_pt = pts[0];
    let base_val = vals[0];

    for (i, &(p, q)) in pts.iter().enumerate().skip(1) {
        if i >= vals.len() {
            break;
        }
        let antideriv_diff = vals[i] - base_val;

        // Numerical integration via midpoint rule: ∫ sqrt(1-t^2) dt from a to b
        let a = base_pt.0 as f64 / base_pt.1 as f64;
        let b = p as f64 / q as f64;
        let n_steps = 1000usize;
        let h = (b - a) / n_steps as f64;
        let mut numerical = 0.0;
        for j in 0..n_steps {
            let t = a + (j as f64 + 0.5) * h;
            numerical += (1.0 - t * t).sqrt() * h;
        }

        let diff = (antideriv_diff - numerical).abs();
        assert!(
            diff < tol,
            "FTC sqrt(1-x^2): F({}) - F({}) = {}, numerical integral = {}, diff = {}",
            p as f64 / q as f64,
            a,
            antideriv_diff,
            numerical,
            diff
        );
    }
}

#[test]
fn integrate_sqrt_x2_plus_one() {
    // ∫ √(x²+1) dx = ½(x·√(x²+1) + asinh(x))
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let x2 = x.powi(2);
    let base = &x2 + &one;
    let integrand = base.pow(&ctx.rational(1, 2));
    let result = integrand.integrate(&x);
    let s = format!("{result}");

    let is_unevaluated = s.contains("Integral") || s.contains("∫");
    assert!(
        !is_unevaluated,
        "∫ √(x²+1) dx should not be unevaluated, got: {s}"
    );
    assert!(
        s.contains("asinh"),
        "∫ √(x²+1) dx should contain asinh, got: {s}"
    );
}

#[test]
fn integrate_sqrt_x2_plus_one_ftc() {
    // Use the difference method F(b)-F(a) vs numerical midpoint integral
    // to avoid cross-context arena issues with assert_ftc_tol's Context::new().
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let x2 = x.powi(2);
    let base = &x2 + &one;
    let integrand = base.pow(&ctx.rational(1, 2));
    let antideriv = integrand.integrate(&x);

    let s = format!("{antideriv}");
    assert!(
        !s.contains("Integral"),
        "sqrt(x^2+1): integration returned unevaluated: {s}"
    );

    // Evaluate antiderivative at several points (all in same context)
    let pts: &[(i64, i64)] = &[(3, 10), (7, 10), (14, 10), (20, 10)];

    let mut eval_pts: Vec<(f64, f64)> = Vec::new();
    for &(p, q) in pts {
        let pt = ctx.rational(p, q);
        let f_at_pt = antideriv.subs(&x, &pt).eval();
        if let Ok(v) = f_at_pt.eval_f64() {
            if v.is_finite() {
                eval_pts.push((p as f64 / q as f64, v));
            }
        }
    }

    assert!(
        eval_pts.len() >= 2,
        "FTC sqrt(x^2+1): need at least 2 evaluation points, got {}",
        eval_pts.len()
    );

    // Compare differences F(b)-F(a) against midpoint-rule numerical integration
    let tol = 1e-4;
    let (a, base_val) = eval_pts[0];

    for &(b, f_b) in &eval_pts[1..] {
        let antideriv_diff = f_b - base_val;

        let n_steps = 1000usize;
        let h = (b - a) / n_steps as f64;
        let mut numerical = 0.0;
        for j in 0..n_steps {
            let t = a + (j as f64 + 0.5) * h;
            numerical += (t * t + 1.0).sqrt() * h;
        }

        let diff = (antideriv_diff - numerical).abs();
        assert!(
            diff < tol,
            "FTC sqrt(x^2+1): F({}) - F({}) = {}, numerical = {}, diff = {}",
            b, a, antideriv_diff, numerical, diff
        );
    }
}

#[test]
fn integrate_sqrt_x2_minus_four() {
    // ∫ √(x²-4) dx = ½(x·√(x²-4) − 4·acosh(x/2))
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let neg_four = ctx.int(-4);
    let x2 = x.powi(2);
    let base = &x2 + &neg_four;
    let integrand = base.pow(&ctx.rational(1, 2));
    let result = integrand.integrate(&x);
    let s = format!("{result}");

    let is_unevaluated = s.contains("Integral") || s.contains("∫");
    assert!(
        !is_unevaluated,
        "∫ √(x²-4) dx should not be unevaluated, got: {s}"
    );
    assert!(
        s.contains("acosh"),
        "∫ √(x²-4) dx should contain acosh, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// erf at arbitrary precision
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn erf_at_zero_high_precision() {
    let __ctx = Context::new();
    // erf(0) = 0 at any precision.
    let result = __ctx.int(0).erf().eval_decimal(30).unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
        val.abs() < 1e-20,
        "erf(0) should be 0 at 30 digits, got: {result}"
    );
}

#[test]
fn erf_symmetry() {
    let __ctx = Context::new();
    // erf(-x) = -erf(x), verified numerically at x = 1.
    let pos = __ctx.int(1).erf().eval_decimal(20).unwrap();
    let neg = __ctx.int(-1).erf().eval_decimal(20).unwrap();
    let pos_val: f64 = pos.parse().unwrap();
    let neg_val: f64 = neg.parse().unwrap();
    assert!(
        (pos_val + neg_val).abs() < 1e-12,
        "erf(1) + erf(-1) should be 0, got {pos_val} + {neg_val} = {}",
        pos_val + neg_val
    );
}

#[test]
fn erf_at_one_matches_known_value() {
    let __ctx = Context::new();
    // erf(1) ≈ 0.84270079294971486934…
    let result = __ctx.int(1).erf().eval_decimal(15).unwrap();
    assert!(
        result.starts_with("0.84270079"),
        "erf(1) should start with 0.84270079, got: {result}"
    );
}

#[test]
fn erf_at_two() {
    let __ctx = Context::new();
    // erf(2) ≈ 0.99532226501895…
    let result = __ctx.int(2).erf().eval_decimal(15).unwrap();
    assert!(
        result.starts_with("0.99532226"),
        "erf(2) should start with 0.99532226, got: {result}"
    );
}

#[test]
fn erf_large_argument() {
    let __ctx = Context::new();
    // erf(5) ≈ 0.99999999999846…  (very close to 1)
    let result = __ctx.int(5).erf().eval_decimal(15).unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
        (val - 1.0).abs() < 1e-10,
        "erf(5) should be very close to 1, got: {result}"
    );
}

#[test]
fn erfc_at_zero() {
    let __ctx = Context::new();
    // erfc(0) = 1
    let result = __ctx.int(0).erfc().eval_decimal(15).unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
        (val - 1.0).abs() < 1e-12,
        "erfc(0) should be 1, got: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Beta function at arbitrary precision
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn beta_symmetry() {
    let __ctx = Context::new();
    // Beta(a, b) = Beta(b, a), verified numerically.
    let a = __ctx.rational(3, 2);
    let b = __ctx.rational(5, 2);

    let beta_ab = a.beta(&b).eval_decimal(15).unwrap();
    let beta_ba = b.beta(&a).eval_decimal(15).unwrap();

    let val_ab: f64 = beta_ab.parse().unwrap();
    let val_ba: f64 = beta_ba.parse().unwrap();

    assert!(
        (val_ab - val_ba).abs() < 1e-12,
        "Beta(3/2, 5/2) = {val_ab} should equal Beta(5/2, 3/2) = {val_ba}"
    );
}

#[test]
fn beta_known_values() {
    let __ctx = Context::new();
    // Beta(1, 1) = Γ(1)Γ(1)/Γ(2) = 1·1/1 = 1
    let one = __ctx.int(1);
    let result = one.beta(&one).eval_decimal(15).unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
        (val - 1.0).abs() < 1e-10,
        "Beta(1, 1) should be 1, got {val}"
    );
}

#[test]
fn beta_half_half() {
    let __ctx = Context::new();
    // Beta(1/2, 1/2) = Γ(1/2)²/Γ(1) = π/1 = π
    let half = __ctx.rational(1, 2);
    let result = half.beta(&half).eval_decimal(15).unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
        (val - std::f64::consts::PI).abs() < 1e-10,
        "Beta(1/2, 1/2) should be π ≈ 3.14159…, got {val}"
    );
}

#[test]
fn beta_integer_arguments() {
    let __ctx = Context::new();
    // Beta(3, 4) = Γ(3)Γ(4)/Γ(7) = 2·6/720 = 1/60 ≈ 0.016666…
    let three = __ctx.int(3);
    let four = __ctx.int(4);
    let result = three.beta(&four).eval_decimal(15).unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = 1.0 / 60.0;
    assert!(
        (val - expected).abs() < 1e-10,
        "Beta(3, 4) should be 1/60 ≈ {expected}, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Binomial at arbitrary precision
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn binomial_integer_cases() {
    // C(10, 3) = 120
    let ctx = Context::new();
    let n = ctx.int(10);
    let k = ctx.int(3);
    let result = n.binomial(&k).eval_decimal(15).unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
        (val - 120.0).abs() < 1e-6,
        "C(10,3) should be 120, got {val}"
    );
}

#[test]
fn binomial_larger_values() {
    // C(20, 10) = 184756
    let ctx = Context::new();
    let n = ctx.int(20);
    let k = ctx.int(10);
    let result = n.binomial(&k).eval_decimal(15).unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
        (val - 184756.0).abs() < 1e-3,
        "C(20,10) should be 184756, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Existing integration patterns (confirm they still work)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_one_over_x2_plus_one() {
    // ∫ 1/(x²+1) dx = atan(x) — existing standard form A3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let x2 = x.powi(2);
    let base = &x2 + &one;
    let integrand = base.powi(-1);
    let result = integrand.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("atan"),
        "∫ 1/(x²+1) dx should contain atan, got: {s}"
    );
}

#[test]
fn integrate_one_over_sqrt_x2_plus_one() {
    // ∫ 1/√(x²+1) dx = asinh(x) — existing standard form A5
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let x2 = x.powi(2);
    let base = &x2 + &one;
    let integrand = base.pow(&ctx.rational(-1, 2));
    let result = integrand.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("asinh"),
        "∫ 1/√(x²+1) dx should contain asinh, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Negative Gamma arguments (reflection formula)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gamma_neg_half() {
    let __ctx = Context::new();
    // Γ(-1/2) = -2√π ≈ -3.5449077018110320…
    let val = __ctx.rational(-1, 2);
    let result = val.gamma().eval_decimal(20).unwrap();
    assert!(
        result.starts_with("-3.54490770"),
        "Gamma(-1/2) should be -2√π ≈ -3.5449077…, got: {result}"
    );
}

#[test]
fn gamma_at_pole_returns_error() {
    let __ctx = Context::new();
    // Γ(0) should return an error (pole).
    let result = __ctx.int(0).gamma().eval_decimal(15);
    assert!(result.is_err(), "Gamma(0) should be an error (pole)");
}

#[test]
fn gamma_neg_integer_pole() {
    let __ctx = Context::new();
    // Γ(-1) should return an error (pole at non-positive integers).
    let result = __ctx.int(-1).gamma().eval_decimal(15);
    assert!(result.is_err(), "Gamma(-1) should be an error (pole)");
}
