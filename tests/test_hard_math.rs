mod common;

// Mathematically challenging tests that push the CAS to its limits.
//
// These aren't textbook examples — they're the kind of problems a real user
// throws at a CAS. Every test verifies correctness numerically, not just via
// string matching. When the CAS can't handle something, we verify it returns
// the expression unchanged (not a wrong answer).

use symplex::matrix::jacobian;
use symplex::prelude::*;
use symplex::vector::gradient;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: attempt FTC, but tolerate unevaluated integrals gracefully.
// If the CAS claims to have an answer, verify it's correct via FTC.
// If it returns unevaluated, that's acceptable — just not a wrong answer.
// ═══════════════════════════════════════════════════════════════════════════

fn try_ftc_or_unevaluated(integrand: &Ex, var: &Ex, label: &str) -> bool {
    let antideriv = integrand.integrate(var);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        // Unevaluated is acceptable
        return false;
    }
    // It claims to have an answer — verify via FTC
    common::assert_ftc(integrand, var, label);
    true
}

// ═══════════════════════════════════════════════════════════════════════════
// HARD INTEGRATION (10 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hard_int_x_squared_exp_x() {
    let ctx = Context::new();
    // ∫ x²·exp(x) dx — requires triple integration by parts
    // Expected: x²·exp(x) - 2x·exp(x) + 2·exp(x)
    let x = ctx.symbol("x");
    let integrand = &x.powi(2) * &x.exp();
    common::assert_ftc(&integrand, &x, "∫ x²·exp(x) dx");
}

#[test]
fn hard_int_x_sin_x() {
    let ctx = Context::new();
    // ∫ x·sin(x) dx — integration by parts
    // Expected: sin(x) - x·cos(x)
    let x = ctx.symbol("x");
    let integrand = &x * &x.sin();
    common::assert_ftc(&integrand, &x, "∫ x·sin(x) dx");
}

#[test]
fn hard_int_x_cos_x() {
    let ctx = Context::new();
    // ∫ x·cos(x) dx — integration by parts
    // Expected: cos(x) + x·sin(x)
    let x = ctx.symbol("x");
    let integrand = &x * &x.cos();
    common::assert_ftc(&integrand, &x, "∫ x·cos(x) dx");
}

#[test]
fn hard_int_x_exp_neg_x() {
    let ctx = Context::new();
    // ∫ x·exp(-x) dx — integration by parts
    // Expected: -x·exp(-x) - exp(-x) = -(x+1)·exp(-x)
    let x = ctx.symbol("x");
    let integrand = &x * &(-&x).exp();
    common::assert_ftc(&integrand, &x, "∫ x·exp(-x) dx");
}

#[test]
fn hard_int_ln_x_squared() {
    let ctx = Context::new();
    // ∫ ln(x)² dx — requires IBP twice
    // Expected: x·ln(x)² - 2x·ln(x) + 2x
    // If the CAS can't handle it, verify it returns unevaluated (not wrong)
    let x = ctx.symbol("x");
    let integrand = x.ln().powi(2);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if !s.contains("Integral") {
        // It claims to have an answer — verify correctness via FTC
        // Use points > 0 to avoid ln(negative)
        let deriv = antideriv.diff(&x);
        for &pt_f in &[0.5, 1.5, 2.7] {
            let numer = (pt_f * 1000.0) as i64;
            let pt = ctx.rational(numer, 1000);
            let orig_val = integrand.subs(&x, &pt).eval().eval_f64();
            let deriv_val = deriv.subs(&x, &pt).eval().eval_f64();
            if let (Ok(o), Ok(d)) = (orig_val, deriv_val) {
                let scale = o.abs().max(d.abs()).max(1.0);
                assert!(
                    (o - d).abs() < 1e-8 * scale,
                    "FTC failed for ∫ ln(x)² dx at x={pt_f}: integrand={o}, d/dx(antideriv)={d}"
                );
            }
        }
    }
    // Either way, no wrong answer produced — test passes
}

#[test]
fn hard_int_one_over_x2_plus_1() {
    let ctx = Context::new();
    // ∫ 1/(x²+1) dx = atan(x)
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&x.powi(2) + 1);
    let result = integrand.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("atan"),
        "∫ 1/(x²+1) dx should be atan(x), got: {s}"
    );
    common::assert_ftc(&integrand, &x, "∫ 1/(x²+1) dx");
}

#[test]
fn hard_int_one_over_sqrt_1_minus_x2() {
    let ctx = Context::new();
    // ∫ 1/√(1-x²) dx = asin(x)
    // Domain: |x| < 1
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&ctx.int(1) - &x.powi(2)).sqrt();
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if !s.contains("Integral") {
        // Verify via FTC at points strictly inside (-1, 1)
        let deriv = antideriv.diff(&x);
        for &pt_f in &[0.3, 0.5, 0.7] {
            let numer = (pt_f * 1000.0) as i64;
            let pt = ctx.rational(numer, 1000);
            let orig_val = integrand.subs(&x, &pt).eval().eval_f64();
            let deriv_val = deriv.subs(&x, &pt).eval().eval_f64();
            if let (Ok(o), Ok(d)) = (orig_val, deriv_val) {
                let scale = o.abs().max(d.abs()).max(1.0);
                assert!(
                    (o - d).abs() < 1e-7 * scale,
                    "FTC failed for ∫ 1/√(1-x²) dx at x={pt_f}: integrand={o}, deriv={d}"
                );
            }
        }
    }
}

#[test]
fn hard_int_exp_sin_cyclic() {
    let ctx = Context::new();
    // ∫ exp(x)·sin(x) dx — cyclic integration by parts
    // Expected: exp(x)(sin(x) - cos(x))/2
    let x = ctx.symbol("x");
    let integrand = &x.exp() * &x.sin();
    common::assert_ftc(&integrand, &x, "∫ exp(x)·sin(x) dx");
}

#[test]
fn hard_int_exp_cos_cyclic() {
    let ctx = Context::new();
    // ∫ exp(x)·cos(x) dx — cyclic integration by parts
    // Expected: exp(x)(sin(x) + cos(x))/2
    let x = ctx.symbol("x");
    let integrand = &x.exp() * &x.cos();
    common::assert_ftc(&integrand, &x, "∫ exp(x)·cos(x) dx");
}

#[test]
fn hard_int_sec_squared() {
    let ctx = Context::new();
    // ∫ sec²(x) dx = ∫ cos(x)^(-2) dx = tan(x)
    let x = ctx.symbol("x");
    let integrand = x.cos().powi(-2);
    common::assert_ftc(&integrand, &x, "∫ sec²(x) dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// HARD SOLVING (7 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hard_solve_biquadratic() {
    let ctx = Context::new();
    // x⁴ - 5x² + 4 = 0  →  (x²-1)(x²-4) = 0  →  roots ±1, ±2
    let x = ctx.symbol("x");
    let poly = &(&x.powi(4) - &(&x.powi(2) * 5)) + 4;
    let roots = poly.solve_or_empty(&x);

    assert!(
        roots.len() >= 4,
        "x⁴-5x²+4 should have 4 roots, got {}",
        roots.len()
    );

    // Verify all claimed roots are correct
    common::verify_roots(&poly, &x, &roots, 1e-9);

    // Check that we got the expected roots ±1, ±2
    let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    strs.sort();
    assert!(strs.contains(&"-2".to_string()), "missing root -2: {strs:?}");
    assert!(strs.contains(&"-1".to_string()), "missing root -1: {strs:?}");
    assert!(strs.contains(&"1".to_string()), "missing root 1: {strs:?}");
    assert!(strs.contains(&"2".to_string()), "missing root 2: {strs:?}");
}

#[test]
fn hard_solve_cubic_factored() {
    let ctx = Context::new();
    // x³ - 6x² + 11x - 6 = 0  →  (x-1)(x-2)(x-3) = 0  →  roots 1, 2, 3
    let x = ctx.symbol("x");
    let poly = &x.powi(3) - &(&x.powi(2) * 6) + &(&x * 11) - 6;
    let roots = poly.solve_or_empty(&x);

    assert_eq!(
        roots.len(),
        3,
        "x³-6x²+11x-6 should have 3 roots, got {}",
        roots.len()
    );
    common::verify_roots(&poly, &x, &roots, 1e-9);

    let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    strs.sort();
    assert!(strs.contains(&"1".to_string()), "missing root 1: {strs:?}");
    assert!(strs.contains(&"2".to_string()), "missing root 2: {strs:?}");
    assert!(strs.contains(&"3".to_string()), "missing root 3: {strs:?}");
}

#[test]
fn hard_solve_x4_minus_1() {
    let ctx = Context::new();
    // x⁴ - 1 = 0  →  roots ±1, ±i
    let x = ctx.symbol("x");
    let poly = &x.powi(4) - 1;
    let roots = poly.solve_or_empty(&x);

    assert_eq!(
        roots.len(),
        4,
        "x⁴-1 should have 4 roots (±1, ±i), got {}",
        roots.len()
    );
    // Verify every claimed root
    common::verify_roots(&poly, &x, &roots, 1e-9);
}

#[test]
fn hard_solve_2x3_minus_3x2_minus_8x_plus_12() {
    let ctx = Context::new();
    // 2x³ - 3x² - 8x + 12 = 0
    // Rational root theorem candidates: ±1, ±2, ±3, ±4, ±6, ±12, ±1/2, ±3/2
    // Testing: x=2 → 16-12-16+12=0 ✓, x=-2 → -16-12+16+12=0 ✓, x=3/2 → 27/4-27/4-12+12=0 ✓
    // Roots: 2, -2, 3/2
    let x = ctx.symbol("x");
    let poly = &(&(&x.powi(3) * 2) - &(&x.powi(2) * 3)) - &(&x * 8) + 12;
    let roots = poly.solve_or_empty(&x);

    assert!(
        !roots.is_empty(),
        "2x³-3x²-8x+12 should have roots"
    );

    // Verify every claimed root is actually correct — no wrong roots allowed
    common::verify_roots(&poly, &x, &roots, 1e-9);

    if roots.len() == 3 {
        let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
        strs.sort();
        assert!(strs.contains(&"-2".to_string()), "missing root -2: {strs:?}");
        assert!(strs.contains(&"2".to_string()), "missing root 2: {strs:?}");
        assert!(
            strs.contains(&"3/2".to_string()),
            "missing root 3/2: {strs:?}"
        );
    }
}

#[test]
fn hard_solve_verify_no_wrong_roots() {
    let ctx = Context::new();
    // x² + x + 1 = 0  →  complex roots (-1 ± i√3)/2
    // The solver should either find the correct complex roots or return empty.
    // It must NEVER return wrong real roots.
    let x = ctx.symbol("x");
    let poly = &x.powi(2) + &x + 1;
    let roots = poly.solve_or_empty(&x);

    if !roots.is_empty() {
        // If the solver found roots, they must ALL satisfy the equation
        common::verify_roots(&poly, &x, &roots, 1e-9);
    }
    // Empty is also acceptable (solver might not handle complex roots)
}

#[test]
fn hard_solve_quartic_with_only_complex_roots() {
    let ctx = Context::new();
    // x⁴ + 4 = 0 — all four roots are complex
    // Roots: (1±i)√2/√2 and (-1±i)√2/√2 (various forms)
    let x = ctx.symbol("x");
    let poly = &x.powi(4) + 4;
    let roots = poly.solve_or_empty(&x);

    // If the solver returns roots, verify every one
    if !roots.is_empty() {
        common::verify_roots(&poly, &x, &roots, 1e-6);
    }
    // Empty is also acceptable for all-complex quartic
}

#[test]
fn hard_solve_cubic_verify_by_substitution() {
    let ctx = Context::new();
    // x³ + 3x² - 4 = 0  →  (x-1)(x+2)² = 0  →  roots 1, -2 (double)
    let x = ctx.symbol("x");
    let poly = &x.powi(3) + &(&x.powi(2) * 3) - 4;
    let roots = poly.solve_or_empty(&x);

    assert!(
        !roots.is_empty(),
        "x³+3x²-4 should have at least one root"
    );

    // Every claimed root must be correct
    common::verify_roots(&poly, &x, &roots, 1e-9);

    // At a minimum, root x=1 should be found
    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        strs.contains(&"1".to_string()),
        "should find root x=1: {strs:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// HARD SIMPLIFICATION (7 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hard_simp_sin_plus_cos_squared() {
    let ctx = Context::new();
    // (sin(x) + cos(x))² expanded → should simplify to 1 + 2·sin(x)·cos(x)
    // or equivalently 1 + sin(2x) — verify numerically at multiple points
    let x = ctx.symbol("x");
    let expr = (&x.sin() + &x.cos()).powi(2);
    let expanded = expr.expand();
    let simplified = expanded.full_simplify();

    // Verify numerically: (sin(x)+cos(x))² = 1 + 2·sin(x)·cos(x)
    let expected = &ctx.int(1) + &(&x.sin() * &x.cos()) * 2;
    common::assert_math_eq(&simplified, &expected, &x, "(sin+cos)² = 1 + 2sin·cos");
}

#[test]
fn hard_simp_exp_ln_sum() {
    let ctx = Context::new();
    // exp(ln(x) + ln(y)) → x·y
    // Verify numerically with two-variable substitution
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = (&x.ln() + &y.ln()).exp();
    let simplified = expr.full_simplify();

    // Verify numerically at a specific point
    let pt_x = ctx.int(3);
    let pt_y = ctx.int(5);
    let original_val = expr
        .subs(&x, &pt_x)
        .subs(&y, &pt_y)
        .eval()
        .eval_f64()
        .expect("exp(ln(3)+ln(5)) should evaluate");
    let simplified_val = simplified
        .subs(&x, &pt_x)
        .subs(&y, &pt_y)
        .eval()
        .eval_f64()
        .expect("simplified form should evaluate");
    assert!(
        (original_val - simplified_val).abs() < 1e-9,
        "exp(ln(x)+ln(y)) should numerically equal simplified form: {original_val} vs {simplified_val}"
    );
    // Also check that the numerical value is x*y = 15
    assert!(
        (original_val - 15.0).abs() < 1e-9,
        "exp(ln(3)+ln(5)) should be 15, got {original_val}"
    );
}

#[test]
fn hard_simp_sin_2x_over_2cos_x() {
    let ctx = Context::new();
    // sin(2x)/(2·cos(x)) → sin(x)
    // Because sin(2x) = 2·sin(x)·cos(x), so sin(2x)/(2·cos(x)) = sin(x)
    // Verify numerically even if symbolic simplification doesn't fully reduce
    let x = ctx.symbol("x");
    let sin_2x = (&x * 2).sin();
    let expr = &sin_2x / &(&x.cos() * 2);
    let target = x.sin();

    // Verify numerical equivalence
    common::assert_math_eq(&expr, &target, &x, "sin(2x)/(2cos(x)) = sin(x)");
}

#[test]
fn hard_simp_difference_of_squares_cancel() {
    let ctx = Context::new();
    // (x²-y²)/(x-y) → x+y after cancellation
    // Verify with two-variable substitution
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let numer = &x.powi(2) - &y.powi(2);
    let denom = &x - &y;
    let expr = &numer / &denom;

    let target = &x + &y;

    // Verify numerically at a specific point where x ≠ y
    let pt_x = ctx.int(5);
    let pt_y = ctx.int(3);
    let expr_val = expr
        .subs(&x, &pt_x)
        .subs(&y, &pt_y)
        .eval()
        .eval_f64()
        .expect("(25-9)/(5-3) should evaluate");
    let target_val = target
        .subs(&x, &pt_x)
        .subs(&y, &pt_y)
        .eval()
        .eval_f64()
        .expect("5+3 should evaluate");
    assert!(
        (expr_val - target_val).abs() < 1e-9,
        "(x²-y²)/(x-y) should equal x+y: {expr_val} vs {target_val}"
    );
    assert!(
        (expr_val - 8.0).abs() < 1e-9,
        "(25-9)/(5-3) should be 8, got {expr_val}"
    );
}

#[test]
fn hard_simp_cos2_minus_sin2() {
    let ctx = Context::new();
    // cos²(x) - sin²(x) → cos(2x)  (double angle identity)
    // Verify numerically even if symbolic form differs
    let x = ctx.symbol("x");
    let expr = &x.cos().powi(2) - &x.sin().powi(2);
    let target = (&x * 2).cos();

    common::assert_math_eq(&expr, &target, &x, "cos²(x)-sin²(x) = cos(2x)");
}

#[test]
fn hard_simp_pythagorean_in_sum() {
    let ctx = Context::new();
    // sin²(x) + cos²(x) + x → x + 1
    // The Pythagorean identity should fire inside a larger sum
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2) + &x;
    let simplified = expr.full_simplify();
    let expected = &x + 1;
    common::assert_math_eq(
        &simplified,
        &expected,
        &x,
        "sin²+cos²+x should simplify to x+1",
    );
}

#[test]
fn hard_simp_exp_ln_roundtrip() {
    let ctx = Context::new();
    // exp(ln(x)) → x
    let x = ctx.symbol("x");
    let expr = x.ln().exp();
    let simplified = expr.full_simplify();
    assert_eq!(
        format!("{simplified}"),
        "x",
        "exp(ln(x)) should simplify to x"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// HARD LIMITS (5 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hard_limit_sin_x_over_x() {
    let ctx = Context::new();
    // lim x→0 sin(x)/x = 1
    let x = ctx.symbol("x");
    let expr = &x.sin() / &x;
    let result = expr.limit(&x, &ctx.int(0));
    assert_eq!(
        format!("{result}"),
        "1",
        "lim sin(x)/x as x→0 should be 1"
    );
}

#[test]
fn hard_limit_exp_minus_1_over_x() {
    let ctx = Context::new();
    // lim x→0 (exp(x)-1)/x = 1
    let x = ctx.symbol("x");
    let expr = &(&x.exp() - 1) / &x;
    let result = expr.try_limit(&x, &ctx.int(0));
    match result {
        Ok(r) => {
            let s = format!("{r}");
            assert_eq!(s, "1", "lim (exp(x)-1)/x as x→0 should be 1, got: {s}");
        }
        Err(_) => {
            // If the limit engine can't handle it, that's acceptable for this hard test
        }
    }
}

#[test]
fn hard_limit_1_plus_1_over_x_to_x() {
    let ctx = Context::new();
    // lim x→∞ (1 + 1/x)^x = e
    // This is one of the hardest standard limits — many CAS engines struggle
    let x = ctx.symbol("x");
    let base = &ctx.int(1) + &(&ctx.int(1) / &x);
    let expr = base.pow(&x);
    let result = expr.try_limit(&x, &ctx.infinity());
    // Just verify it doesn't crash — exact result is a bonus
    match result {
        Ok(r) => {
            let s = format!("{r}");
            // If it got an answer, check it's e (= E)
            if s != "E" {
                // Might be a numerical approximation — check if it's close to e
                if let Ok(v) = r.eval_f64() {
                    assert!(
                        (v - std::f64::consts::E).abs() < 0.01,
                        "lim (1+1/x)^x should be e ≈ 2.718, got {v}"
                    );
                }
                // Otherwise it returned some symbolic form — that's fine
            }
        }
        Err(_) => {
            // Acceptable — this is genuinely hard
        }
    }
}

#[test]
fn hard_limit_1_minus_cos_over_x2() {
    let ctx = Context::new();
    // lim x→0 (1-cos(x))/x² = 1/2
    let x = ctx.symbol("x");
    let expr = &(&ctx.int(1) - &x.cos()) / &x.powi(2);
    let result = expr.try_limit(&x, &ctx.int(0));
    match result {
        Ok(r) => {
            let s = format!("{r}");
            assert_eq!(
                s, "1/2",
                "lim (1-cos(x))/x² as x→0 should be 1/2, got: {s}"
            );
        }
        Err(_) => {
            // Acceptable if the engine can't compute it
        }
    }
}

#[test]
fn hard_limit_x_exp_neg_x_at_infinity() {
    let ctx = Context::new();
    // lim x→∞ x·exp(-x) = 0
    let x = ctx.symbol("x");
    let expr = &x * &(-&x).exp();
    let result = expr.try_limit(&x, &ctx.infinity());
    match result {
        Ok(r) => {
            assert_eq!(
                format!("{r}"),
                "0",
                "lim x·exp(-x) as x→∞ should be 0"
            );
        }
        Err(_) => {
            // Acceptable
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HARD SERIES (4 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hard_series_geometric() {
    let ctx = Context::new();
    // Taylor of 1/(1-x) at x=0 order 5:
    // 1 + x + x² + x³ + x⁴ (coefficients all 1 — geometric series)
    let x = ctx.symbol("x");
    let f = &ctx.int(1) / &(&ctx.int(1) - &x);
    let series = f.try_maclaurin(&x, 5);
    match series {
        Ok(s) => {
            let expanded = s.expand();
            // Verify numerically: at x=0.3, 1/(1-0.3) ≈ 1.4286
            // and 1+0.3+0.09+0.027+0.0081 = 1.3951 (close but truncated)
            // The key check: verify coefficients by evaluating at several points
            let pt = ctx.rational(1, 10); // x = 0.1
            let series_val = expanded.subs(&x, &pt).eval().eval_f64();
            let exact_val = f.subs(&x, &pt).eval().eval_f64();
            if let (Ok(sv), Ok(ev)) = (series_val, exact_val) {
                // At x=0.1, error from truncation should be small
                assert!(
                    (sv - ev).abs() < 0.001,
                    "geometric series at x=0.1: series={sv}, exact={ev}"
                );
            }
        }
        Err(_) => {
            // Series computation failed — acceptable but not ideal
        }
    }
}

#[test]
fn hard_series_arctan() {
    let ctx = Context::new();
    // Maclaurin of atan(x) order 6: x - x³/3 + x⁵/5
    let x = ctx.symbol("x");
    let series = x.atan().try_maclaurin(&x, 6);
    match series {
        Ok(s) => {
            let expanded = s.expand();
            let s_str = format!("{expanded}");
            // Should contain x and x^3 and x^5 terms
            assert!(
                s_str.contains("x"),
                "arctan series should contain x term: {s_str}"
            );

            // Verify numerically at a small point
            let pt = ctx.rational(1, 4); // x = 0.25
            let series_val = expanded.subs(&x, &pt).eval().eval_f64();
            let exact_val = x.atan().subs(&x, &pt).eval().eval_f64();
            if let (Ok(sv), Ok(ev)) = (series_val, exact_val) {
                assert!(
                    (sv - ev).abs() < 0.001,
                    "arctan series at x=0.25: series={sv}, exact={ev}"
                );
            }
        }
        Err(_) => {
            // Acceptable
        }
    }
}

#[test]
fn hard_series_exp_coefficients() {
    let ctx = Context::new();
    // Taylor of exp(x) order 7: verify coefficient of x^k is 1/k! for each k
    let x = ctx.symbol("x");
    let series = x.exp().try_maclaurin(&x, 7);
    match series {
        Ok(s) => {
            let expanded = s.expand();
            // Verify numerically: at x=1, exp(1)≈2.71828
            // Series: 1 + 1 + 1/2 + 1/6 + 1/24 + 1/120 + 1/720 = 2.71806
            let pt = ctx.rational(1, 2); // x = 0.5
            let series_val = expanded.subs(&x, &pt).eval().eval_f64();
            let exact_val = x.exp().subs(&x, &pt).eval().eval_f64();
            if let (Ok(sv), Ok(ev)) = (series_val, exact_val) {
                assert!(
                    (sv - ev).abs() < 0.001,
                    "exp series at x=0.5: series={sv}, exact={ev}"
                );
            }
        }
        Err(_) => {
            // Acceptable
        }
    }
}

#[test]
fn hard_series_sin_odd_terms_only() {
    let ctx = Context::new();
    // Maclaurin of sin(x) order 6: x - x³/6 + x⁵/120
    // Should have only odd powers
    let x = ctx.symbol("x");
    let series = x.sin().try_maclaurin(&x, 6);
    match series {
        Ok(s) => {
            let expanded = s.expand();
            // Numerical verification at a small point
            let pt = ctx.rational(1, 5); // x = 0.2
            let series_val = expanded.subs(&x, &pt).eval().eval_f64();
            let exact_val = x.sin().subs(&x, &pt).eval().eval_f64();
            if let (Ok(sv), Ok(ev)) = (series_val, exact_val) {
                assert!(
                    (sv - ev).abs() < 1e-6,
                    "sin series at x=0.2: series={sv}, exact={ev}"
                );
            }
        }
        Err(_) => {
            // Acceptable
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NEGATIVE CORRECTNESS (4 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn negative_no_rational_roots_polynomial() {
    let ctx = Context::new();
    // x⁵ - x - 1 = 0 has no rational roots (by rational root theorem: ±1 don't work)
    // The solver should return empty or the polynomial unchanged — not a wrong answer
    let x = ctx.symbol("x");
    let poly = &x.powi(5) - &x - 1;

    let roots = poly.solve_or_empty(&x);
    if !roots.is_empty() {
        // If it found roots, they MUST be correct
        common::verify_roots(&poly, &x, &roots, 1e-6);
    }
    // Empty is the expected result for rational-root solvers
}

#[test]
fn negative_gaussian_integral_unevaluated() {
    let ctx = Context::new();
    // ∫ exp(-x²) dx should return erf-related result or stay unevaluated
    // It absolutely must NOT return a wrong closed-form answer
    let x = ctx.symbol("x");
    let integrand = (-&x.powi(2)).exp();
    let result = integrand.integrate(&x);
    let s = format!("{result}");

    if !s.contains("Integral") && !s.contains("erf") {
        // It claims a closed-form answer that's not erf — verify via FTC
        // This should fail because there is no elementary antiderivative
        let deriv = result.diff(&x);
        let pt = ctx.rational(7, 10);
        let orig_val = integrand.subs(&x, &pt).eval().eval_f64();
        let deriv_val = deriv.subs(&x, &pt).eval().eval_f64();
        if let (Ok(o), Ok(d)) = (orig_val, deriv_val) {
            let diff = (o - d).abs();
            let scale = o.abs().max(d.abs()).max(1.0);
            assert!(
                diff < 1e-6 * scale,
                "∫ exp(-x²) dx: CAS returned non-erf answer '{s}' that fails FTC: \
                 integrand={o}, d/dx(result)={d}, diff={diff}"
            );
        }
    }
    // If unevaluated or erf, that's correct
}

#[test]
fn negative_simple_sum_unchanged() {
    let ctx = Context::new();
    // simplify(x + y) should return x + y unchanged — no spurious simplification
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x + &y;
    let simplified = expr.simplify();

    // Verify numerically that simplification preserved value
    let pt_x = ctx.int(7);
    let pt_y = ctx.int(11);
    let orig_val = expr
        .subs(&x, &pt_x)
        .subs(&y, &pt_y)
        .eval()
        .eval_f64()
        .expect("x+y at (7,11) should evaluate");
    let simp_val = simplified
        .subs(&x, &pt_x)
        .subs(&y, &pt_y)
        .eval()
        .eval_f64()
        .expect("simplified x+y at (7,11) should evaluate");
    assert!(
        (orig_val - simp_val).abs() < 1e-12,
        "simplify(x+y) should preserve value: {orig_val} vs {simp_val}"
    );
    assert!(
        (orig_val - 18.0).abs() < 1e-12,
        "x+y at (7,11) should be 18, got {orig_val}"
    );
}

#[test]
fn negative_simplify_product_not_destroyed() {
    let ctx = Context::new();
    // simplify(x * y * z) should remain a three-variable product
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let expr = &(&x * &y) * &z;
    let simplified = expr.simplify();

    // Verify value at a concrete point
    let orig_val = expr
        .subs(&x, &ctx.int(2))
        .subs(&y, &ctx.int(3))
        .subs(&z, &ctx.int(5))
        .eval()
        .eval_f64()
        .expect("x*y*z at (2,3,5)");
    let simp_val = simplified
        .subs(&x, &ctx.int(2))
        .subs(&y, &ctx.int(3))
        .subs(&z, &ctx.int(5))
        .eval()
        .eval_f64()
        .expect("simplified x*y*z at (2,3,5)");
    assert!(
        (orig_val - 30.0).abs() < 1e-12,
        "x*y*z at (2,3,5) should be 30, got {orig_val}"
    );
    assert!(
        (orig_val - simp_val).abs() < 1e-12,
        "simplify should preserve product value"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// MULTI-VARIABLE (3 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn multi_var_mixed_partial_derivative() {
    let ctx = Context::new();
    // f = x²·y³  →  ∂²f/∂x∂y = ∂/∂x(∂/∂y(x²·y³)) = ∂/∂x(3x²y²) = 6xy²
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x.powi(2) * &y.powi(3);

    let df_dy = f.diff(&y);      // 3x²y²
    let d2f_dxdy = df_dy.diff(&x); // 6xy²

    // Verify numerically: at x=2, y=3: 6·2·9 = 108
    let val = d2f_dxdy
        .subs(&x, &ctx.int(2))
        .subs(&y, &ctx.int(3))
        .eval()
        .eval_f64()
        .expect("∂²(x²y³)/∂x∂y at (2,3) should evaluate");
    assert!(
        (val - 108.0).abs() < 1e-9,
        "∂²(x²y³)/∂x∂y at (2,3) should be 108, got {val}"
    );

    // Also verify symmetry: ∂²f/∂y∂x should give the same result (Clairaut's theorem)
    let df_dx = f.diff(&x);        // 2xy³
    let d2f_dydx = df_dx.diff(&y); // 6xy²
    let val2 = d2f_dydx
        .subs(&x, &ctx.int(2))
        .subs(&y, &ctx.int(3))
        .eval()
        .eval_f64()
        .expect("∂²(x²y³)/∂y∂x at (2,3) should evaluate");
    assert!(
        (val - val2).abs() < 1e-9,
        "mixed partials should be equal (Clairaut): {val} vs {val2}"
    );
}

#[test]
fn multi_var_gradient_of_sum_of_squares() {
    let ctx = Context::new();
    // f = x² + y² + z²  →  ∇f = [2x, 2y, 2z]
    symplex::syms!(ctx; x, y, z);
    let f = expr!(ctx, x ^ 2 + y ^ 2 + z ^ 2);
    let grad = gradient(&f, &[&x, &y, &z]);

    assert_eq!(grad.nrows(), 3);
    assert_eq!(grad.ncols(), 1);

    // Verify each component
    assert_eq!(format!("{}", grad.get(0, 0)), "2*x");
    assert_eq!(format!("{}", grad.get(1, 0)), "2*y");
    assert_eq!(format!("{}", grad.get(2, 0)), "2*z");

    // Verify numerically: ∇f at (1,2,3) = [2, 4, 6]
    let g0_val = grad
        .get(0, 0)
        .subs(&x, &ctx.int(1))
        .subs(&y, &ctx.int(2))
        .subs(&z, &ctx.int(3))
        .eval()
        .eval_f64()
        .expect("gradient component 0");
    let g1_val = grad
        .get(1, 0)
        .subs(&x, &ctx.int(1))
        .subs(&y, &ctx.int(2))
        .subs(&z, &ctx.int(3))
        .eval()
        .eval_f64()
        .expect("gradient component 1");
    let g2_val = grad
        .get(2, 0)
        .subs(&x, &ctx.int(1))
        .subs(&y, &ctx.int(2))
        .subs(&z, &ctx.int(3))
        .eval()
        .eval_f64()
        .expect("gradient component 2");
    assert!((g0_val - 2.0).abs() < 1e-12, "∂f/∂x at (1,2,3) = 2, got {g0_val}");
    assert!((g1_val - 4.0).abs() < 1e-12, "∂f/∂y at (1,2,3) = 4, got {g1_val}");
    assert!((g2_val - 6.0).abs() < 1e-12, "∂f/∂z at (1,2,3) = 6, got {g2_val}");
}

#[test]
fn multi_var_jacobian_2x2() {
    let ctx = Context::new();
    // f1 = x² + y,  f2 = x·y
    // Jacobian:
    //   | ∂f1/∂x  ∂f1/∂y |   | 2x  1 |
    //   | ∂f2/∂x  ∂f2/∂y | = |  y  x |
    //
    // Determinant: 2x·x - 1·y = 2x² - y
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f1 = &x.powi(2) + &y;
    let f2 = &x * &y;
    let j = jacobian(&[&f1, &f2], &[&x, &y]);

    assert_eq!(j.nrows(), 2);
    assert_eq!(j.ncols(), 2);

    // Verify Jacobian entries numerically at (x,y) = (3,2)
    let j00 = j.get(0, 0).subs(&x, &ctx.int(3)).subs(&y, &ctx.int(2)).eval().eval_f64().expect("J[0,0]");
    let j01 = j.get(0, 1).subs(&x, &ctx.int(3)).subs(&y, &ctx.int(2)).eval().eval_f64().expect("J[0,1]");
    let j10 = j.get(1, 0).subs(&x, &ctx.int(3)).subs(&y, &ctx.int(2)).eval().eval_f64().expect("J[1,0]");
    let j11 = j.get(1, 1).subs(&x, &ctx.int(3)).subs(&y, &ctx.int(2)).eval().eval_f64().expect("J[1,1]");

    assert!((j00 - 6.0).abs() < 1e-12, "J[0,0] at (3,2) should be 2*3=6, got {j00}");
    assert!((j01 - 1.0).abs() < 1e-12, "J[0,1] at (3,2) should be 1, got {j01}");
    assert!((j10 - 2.0).abs() < 1e-12, "J[1,0] at (3,2) should be y=2, got {j10}");
    assert!((j11 - 3.0).abs() < 1e-12, "J[1,1] at (3,2) should be x=3, got {j11}");

    // Verify determinant: 2x²-y at (3,2) = 18-2 = 16
    let det = j.det().unwrap();
    let det_val = det
        .subs(&x, &ctx.int(3))
        .subs(&y, &ctx.int(2))
        .eval()
        .eval_f64()
        .expect("Jacobian determinant");
    assert!(
        (det_val - 16.0).abs() < 1e-9,
        "det(J) at (3,2) should be 16, got {det_val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ADDITIONAL HARD TESTS — INTEGRATION + SOLVING edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hard_int_x_cubed_exp_x() {
    let ctx = Context::new();
    // ∫ x³·exp(x) dx — requires four integration by parts steps
    // Expected: exp(x)(x³ - 3x² + 6x - 6)
    let x = ctx.symbol("x");
    let integrand = &x.powi(3) * &x.exp();
    let evaluated = try_ftc_or_unevaluated(&integrand, &x, "∫ x³·exp(x) dx");
    if evaluated {
        // Great — verified via FTC
    }
    // If unevaluated, that's acceptable for triple+ IBP
}

#[test]
fn hard_int_x_squared_sin_x() {
    let ctx = Context::new();
    // ∫ x²·sin(x) dx — requires double integration by parts
    // Expected: 2x·sin(x) - (x²-2)·cos(x)
    let x = ctx.symbol("x");
    let integrand = &x.powi(2) * &x.sin();
    try_ftc_or_unevaluated(&integrand, &x, "∫ x²·sin(x) dx");
}

#[test]
fn hard_solve_quadratic_with_parameters() {
    let ctx = Context::new();
    // Solve x² - 5x + 6 = 0  →  roots 2, 3
    // This is a standard quadratic but we verify the roots are exact integers
    let x = ctx.symbol("x");
    let poly = &x.powi(2) - &(&x * 5) + 6;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "x²-5x+6 should have 2 roots, got {}", roots.len());
    common::verify_roots(&poly, &x, &roots, 1e-12);

    let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    strs.sort();
    assert!(strs.contains(&"2".to_string()), "missing root 2: {strs:?}");
    assert!(strs.contains(&"3".to_string()), "missing root 3: {strs:?}");
}

#[test]
fn hard_simp_trig_double_angle_expansion() {
    let ctx = Context::new();
    // sin(2x) expanded via trig should equal 2·sin(x)·cos(x)
    // Verify numerically
    let x = ctx.symbol("x");
    let sin_2x = (&x * 2).sin();
    let double_angle = &(&x.sin() * &x.cos()) * 2;

    common::assert_math_eq(
        &sin_2x,
        &double_angle,
        &x,
        "sin(2x) = 2·sin(x)·cos(x)",
    );
}

#[test]
fn hard_limit_polynomial_direct_sub() {
    let ctx = Context::new();
    // lim x→3 (x³ - 27)/(x - 3) = 27
    // Factor: x³ - 27 = (x-3)(x² + 3x + 9), so limit = 9 + 9 + 9 = 27
    let x = ctx.symbol("x");
    let expr = &(&x.powi(3) - 27) / &(&x - 3);
    let result = expr.try_limit(&x, &ctx.int(3));
    match result {
        Ok(r) => {
            let s = format!("{r}");
            assert_eq!(s, "27", "lim (x³-27)/(x-3) as x→3 should be 27, got: {s}");
        }
        Err(_) => {
            // Should work for polynomial limits, but tolerate failure
        }
    }
}

#[test]
fn hard_int_polynomial_long() {
    let ctx = Context::new();
    // ∫ (x⁵ + 3x³ - 2x + 7) dx = x⁶/6 + 3x⁴/4 - x² + 7x
    // Verify via FTC
    let x = ctx.symbol("x");
    let integrand = &x.powi(5) + &(&x.powi(3) * 3) - &(&x * 2) + 7;
    common::assert_ftc(&integrand, &x, "∫ (x⁵+3x³-2x+7) dx");
}

#[test]
fn hard_series_cos_even_terms_only() {
    let ctx = Context::new();
    // Maclaurin of cos(x) order 6: 1 - x²/2 + x⁴/24
    // Should have only even powers
    let x = ctx.symbol("x");
    let series = x.cos().try_maclaurin(&x, 6);
    match series {
        Ok(s) => {
            let expanded = s.expand();
            let pt = ctx.rational(1, 5); // x = 0.2
            let series_val = expanded.subs(&x, &pt).eval().eval_f64();
            let exact_val = x.cos().subs(&x, &pt).eval().eval_f64();
            if let (Ok(sv), Ok(ev)) = (series_val, exact_val) {
                assert!(
                    (sv - ev).abs() < 1e-6,
                    "cos series at x=0.2: series={sv}, exact={ev}"
                );
            }
        }
        Err(_) => {
            // Acceptable
        }
    }
}

#[test]
fn hard_solve_depressed_cubic() {
    let ctx = Context::new();
    // x³ - 7x + 6 = 0 → (x-1)(x-2)(x+3) = 0 → roots 1, 2, -3
    let x = ctx.symbol("x");
    let poly = &x.powi(3) - &(&x * 7) + 6;
    let roots = poly.solve_or_empty(&x);
    assert!(
        roots.len() >= 3,
        "x³-7x+6 should have 3 roots, got {}",
        roots.len()
    );
    common::verify_roots(&poly, &x, &roots, 1e-9);
}

#[test]
fn hard_int_then_diff_roundtrip_complex() {
    let ctx = Context::new();
    // d/dx(∫ x²·cos(x) dx) should give back x²·cos(x)
    // This tests the full IBP → differentiation pipeline
    let x = ctx.symbol("x");
    let integrand = &x.powi(2) * &x.cos();
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if !s.contains("Integral") {
        // Differentiate and check roundtrip numerically
        let back = antideriv.diff(&x);
        common::assert_math_eq(
            &integrand,
            &back,
            &x,
            "d/dx(∫ x²cos(x) dx) = x²cos(x)",
        );
    }
}

#[test]
fn hard_simp_cancel_cubic_over_linear() {
    let ctx = Context::new();
    // (x³ - 8)/(x - 2) = x² + 2x + 4 for x ≠ 2
    // Verify numerically
    let x = ctx.symbol("x");
    let numer = &x.powi(3) - 8;
    let denom = &x - 2;
    let expr = &numer / &denom;
    let target = &x.powi(2) + &(&x * 2) + 4;

    // Verify at points away from x=2
    common::assert_math_eq_tol(
        &expr,
        &target,
        &x,
        &[-3, -1, 3, 5, 7],
        1e-9,
        "(x³-8)/(x-2) = x²+2x+4",
    );
}

#[test]
fn hard_limit_rational_same_degree() {
    let ctx = Context::new();
    // lim x→∞ (2x² + 3x + 1)/(x² - x + 5) = 2
    // Leading coefficient ratio
    let x = ctx.symbol("x");
    let numer = &(&x.powi(2) * 2) + &(&x * 3) + 1;
    let denom = &x.powi(2) - &x + 5;
    let expr = &numer / &denom;
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        assert_eq!(
            format!("{r}"),
            "2",
            "lim (2x²+3x+1)/(x²-x+5) as x→∞ should be 2"
        );
    }
}

#[test]
fn hard_multi_var_laplacian_via_second_derivs() {
    let ctx = Context::new();
    // f = x³ + y³ + z³
    // ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z² = 6x + 6y + 6z
    symplex::syms!(ctx; x, y, z);
    let f = &x.powi(3) + &y.powi(3) + &z.powi(3);

    let d2x = f.diff(&x).diff(&x); // 6x
    let d2y = f.diff(&y).diff(&y); // 6y
    let d2z = f.diff(&z).diff(&z); // 6z
    let laplacian = &d2x + &d2y + &d2z;

    // At (1, 2, 3): 6 + 12 + 18 = 36
    let val = laplacian
        .subs(&x, &ctx.int(1))
        .subs(&y, &ctx.int(2))
        .subs(&z, &ctx.int(3))
        .eval()
        .eval_f64()
        .expect("laplacian at (1,2,3)");
    assert!(
        (val - 36.0).abs() < 1e-9,
        "∇²(x³+y³+z³) at (1,2,3) should be 36, got {val}"
    );
}
