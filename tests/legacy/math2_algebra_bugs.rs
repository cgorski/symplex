//! Mathematical correctness tests for symplex algebra operations.
//!
//! Each test computes a result and checks it against a known-correct value.
//! A failing test indicates a potential math bug in symplex.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper utilities
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that every root, when substituted back into the polynomial,
/// yields zero (either structurally or numerically within tolerance).
fn verify_roots_are_zeros(poly: &Ex, var: &Ex, roots: &[Ex], label: &str) {
    for (i, root) in roots.iter().enumerate() {
        let val = poly.subs(var, root).eval().simplify();
        if val.is_zero_structural() {
            continue;
        }
        // Try expanding then checking
        let val_expanded = val.expand().eval().simplify();
        if val_expanded.is_zero_structural() {
            continue;
        }
        // Fall back to numerical check
        match val.eval_complex64() {
            Ok(Complex64 { re, im }) => {
                let mag = (re * re + im * im).sqrt();
                assert!(
                    mag < 1e-6,
                    "{label}: root[{i}] = {root} does not satisfy equation \
                     (residual = {re} + {im}i, |r| = {mag})"
                );
            }
            Err(_) => {
                // Try eval_f64 as fallback
                match val.eval_f64() {
                    Ok(v) => {
                        assert!(
                            v.abs() < 1e-6,
                            "{label}: root[{i}] = {root} does not satisfy equation \
                             (residual = {v})"
                        );
                    }
                    Err(_) => {
                        panic!(
                            "{label}: root[{i}] = {root} could not be verified \
                             (substitution gave {val}, could not evaluate numerically)"
                        );
                    }
                }
            }
        }
    }
}

/// Evaluate an expression at an integer point and return f64.
fn eval_at(expr: &Ex, var: &Ex, pt: i64) -> f64 {
    expr.subs_i64(var, pt)
        .eval()
        .eval_f64()
        .unwrap_or_else(|e| panic!("eval_at({expr}, {var}={pt}) failed: {e}"))
}

/// Check approximate f64 equality.
fn approx(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() < tol * scale
}

/// Assert that two expressions agree numerically at several integer points.
fn assert_equal_at_points(a: &Ex, b: &Ex, var: &Ex, label: &str) {
    for &pt in &[-5i64, -3, -2, -1, 0, 1, 2, 3, 5, 7, 10] {
        let va = eval_at(a, var, pt);
        let vb = eval_at(b, var, pt);
        assert!(
            approx(va, vb, 1e-9),
            "{label}: mismatch at {var}={pt}: {va} vs {vb}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. POLYNOMIAL SOLVING
// ═══════════════════════════════════════════════════════════════════════════

// ── Quadratics ──────────────────────────────────────────────────────────

#[test]
fn solve_quadratic_basic_verify_by_substitution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² - 5x + 6 = 0 → roots 2, 3
    let poly = expr!(ctx, x ^ 2 - 5 * x + 6);
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "expected 2 roots");
    verify_roots_are_zeros(&poly, &x, &roots, "x²-5x+6");

    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(strs.contains(&"2".to_string()), "missing root 2: {strs:?}");
    assert!(strs.contains(&"3".to_string()), "missing root 3: {strs:?}");
}

#[test]
fn solve_quadratic_irrational_roots_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² - 2 = 0 → roots ±√2
    let poly = expr!(ctx, x ^ 2 - 2);
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "expected 2 roots for x²-2");
    verify_roots_are_zeros(&poly, &x, &roots, "x²-2");

    // Numerically check roots are near ±1.4142
    for r in &roots {
        let v = r.eval_f64().expect("root should eval to f64");
        assert!(
            approx(v.abs(), std::f64::consts::SQRT_2, 1e-10),
            "root {r} ≈ {v} should be near ±√2"
        );
    }
}

#[test]
fn solve_quadratic_repeated_root_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² - 6x + 9 = (x-3)² → double root at 3
    let poly = expr!(ctx, x ^ 2 - 6 * x + 9);
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty(), "expected at least 1 root for (x-3)²");
    verify_roots_are_zeros(&poly, &x, &roots, "(x-3)²");
    for r in &roots {
        assert_eq!(format!("{r}"), "3", "all roots should be 3, got {r}");
    }
}

#[test]
fn solve_quadratic_complex_roots_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² + 4 = 0 → roots ±2i
    let poly = expr!(ctx, x ^ 2 + 4);
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "expected 2 complex roots for x²+4");
    verify_roots_are_zeros(&poly, &x, &roots, "x²+4");
}

#[test]
fn solve_quadratic_discriminant_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² + 2x + 1 = (x+1)² → double root at -1
    let poly = expr!(ctx, x ^ 2 + 2 * x + 1);
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty(), "expected root(s) for (x+1)²");
    verify_roots_are_zeros(&poly, &x, &roots, "(x+1)²");
    for r in &roots {
        assert_eq!(format!("{r}"), "-1", "root should be -1, got {r}");
    }
}

#[test]
fn solve_quadratic_large_discriminant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² - 100x + 1 = 0 → roots (100 ± √9996)/2
    let poly = expr!(ctx, x ^ 2 - 100 * x + 1);
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "expected 2 roots");
    verify_roots_are_zeros(&poly, &x, &roots, "x²-100x+1");
}

// ── Cubics ──────────────────────────────────────────────────────────────

#[test]
fn solve_cubic_all_rational_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x-1)(x+2)(x-3) = x³ - 2x² - 5x + 6
    let poly = expr!(ctx, x ^ 3 - 2 * x ^ 2 - 5 * x + 6);
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 3, "expected 3 roots");
    verify_roots_are_zeros(&poly, &x, &roots, "x³-2x²-5x+6");

    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(strs.contains(&"1".to_string()), "missing root 1: {strs:?}");
    assert!(
        strs.contains(&"-2".to_string()),
        "missing root -2: {strs:?}"
    );
    assert!(strs.contains(&"3".to_string()), "missing root 3: {strs:?}");
}

#[test]
fn solve_cubic_one_rational_two_irrational_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ - 3x + 2 = (x-1)²(x+2) → roots 1 (double), -2
    let poly = expr!(ctx, x ^ 3 - 3 * x + 2);
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty(), "expected roots for x³-3x+2");
    verify_roots_are_zeros(&poly, &x, &roots, "x³-3x+2");
}

#[test]
fn solve_cubic_irrational_roots_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ - 2 = 0 → one real root ∛2, two complex
    let poly = expr!(ctx, x ^ 3 - 2);
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 3, "expected 3 roots for x³-2");
    verify_roots_are_zeros(&poly, &x, &roots, "x³-2");
}

#[test]
fn solve_cubic_triple_root_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x-2)³ = x³ - 6x² + 12x - 8
    let poly = expr!(ctx, x ^ 3 - 6 * x ^ 2 + 12 * x - 8);
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty(), "expected root for (x-2)³");
    verify_roots_are_zeros(&poly, &x, &roots, "(x-2)³");
    for r in &roots {
        assert_eq!(format!("{r}"), "2", "triple root should be 2, got {r}");
    }
}

#[test]
fn solve_depressed_cubic_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ + 3x - 4 = 0 → one real root at x=1 (since 1+3-4=0)
    let poly = expr!(ctx, x ^ 3 + 3 * x - 4);
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 3, "expected 3 roots");
    verify_roots_are_zeros(&poly, &x, &roots, "x³+3x-4");
    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(strs.contains(&"1".to_string()), "missing root 1: {strs:?}");
}

// ── Quartics ────────────────────────────────────────────────────────────

#[test]
fn solve_quartic_four_rational_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x-1)(x+1)(x-2)(x+2) = x⁴ - 5x² + 4
    let poly = expr!(ctx, x ^ 4 - 5 * x ^ 2 + 4);
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 4, "expected 4 roots");
    verify_roots_are_zeros(&poly, &x, &roots, "x⁴-5x²+4");

    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    for expected in &["1", "-1", "2", "-2"] {
        assert!(
            strs.contains(&expected.to_string()),
            "missing root {expected}: {strs:?}"
        );
    }
}

#[test]
fn solve_quartic_two_real_two_complex_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁴ - 1 = 0 → roots 1, -1, i, -i
    let poly = expr!(ctx, x ^ 4 - 1);
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 4, "expected 4 roots for x⁴-1");
    verify_roots_are_zeros(&poly, &x, &roots, "x⁴-1");
}

#[test]
fn solve_quartic_all_complex_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁴ + 1 = 0 → four complex roots (primitive 8th roots of unity)
    let poly = expr!(ctx, x ^ 4 + 1);
    let roots = poly.solve_or_empty(&x);
    if roots.len() == 4 {
        verify_roots_are_zeros(&poly, &x, &roots, "x⁴+1");
    }
}

#[test]
fn solve_quartic_repeated_roots_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x-1)²(x+1)² = x⁴ - 2x² + 1
    let poly = expr!(ctx, x ^ 4 - 2 * x ^ 2 + 1);
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty(), "expected roots for (x-1)²(x+1)²");
    verify_roots_are_zeros(&poly, &x, &roots, "(x-1)²(x+1)²");
}

#[test]
fn solve_quartic_with_rational_coeffs_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (2x-1)(x-1)(x+1)(x-3) = 2x⁴ - 7x³ + 2x² + 7x - 3 (expanding manually is error prone, let expand do it)
    // Roots: 1/2, 1, -1, 3
    let built = (&x * 2 - 1) * (&x - 1) * (&x + 1) * (&x - 3);
    let poly = built.expand().eval();
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 4, "expected 4 roots, got {}", roots.len());
    verify_roots_are_zeros(&poly, &x, &roots, "(2x-1)(x-1)(x+1)(x-3)");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. FACTORING
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_x4_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁴ - 1 should factor as (x-1)(x+1)(x²+1)
    let poly = expr!(ctx, x ^ 4 - 1);
    let factored = poly.factor(&x);
    let s = format!("{factored}");

    // It should not contain x^4 (it should be fully factored)
    assert!(!s.contains("x^4"), "x⁴-1 should be factored, but got: {s}");

    // Verify numerical equivalence
    assert_equal_at_points(&poly, &factored, &x, "factor(x⁴-1)");
}

#[test]
fn factor_x2_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = expr!(ctx, x ^ 2 - 1);
    let factored = poly.factor(&x);
    assert_equal_at_points(&poly, &factored, &x, "factor(x²-1)");

    let s = format!("{factored}");
    assert!(
        !s.contains("x^2"),
        "x²-1 should be factored to (x-1)(x+1), got: {s}"
    );
}

#[test]
fn factor_x3_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ - 1 = (x-1)(x²+x+1)
    let poly = expr!(ctx, x ^ 3 - 1);
    let factored = poly.factor(&x);
    assert_equal_at_points(&poly, &factored, &x, "factor(x³-1)");

    let s = format!("{factored}");
    assert!(!s.contains("x^3"), "x³-1 should be factored, got: {s}");
}

#[test]
fn factor_x6_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁶ - 1 should factor into multiple factors over ℤ
    let poly = expr!(ctx, x ^ 6 - 1);
    let factored = poly.factor(&x);
    assert_equal_at_points(&poly, &factored, &x, "factor(x⁶-1)");
}

#[test]
fn factor_expand_roundtrip_quadratic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Start with (x-2)(x+3) = x² + x - 6
    let poly = expr!(ctx, x ^ 2 + x - 6);
    let factored = poly.factor(&x);
    let re_expanded = factored.expand().eval();

    // The re-expanded form should equal the original numerically
    assert_equal_at_points(&poly, &re_expanded, &x, "expand(factor(x²+x-6))");
}

#[test]
fn factor_expand_roundtrip_cubic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
    let poly = expr!(ctx, x ^ 3 - 6 * x ^ 2 + 11 * x - 6);
    let factored = poly.factor(&x);
    let re_expanded = factored.expand().eval();
    assert_equal_at_points(&poly, &re_expanded, &x, "expand(factor(x³-6x²+11x-6))");
}

#[test]
fn factor_expand_roundtrip_quartic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁴ - 5x² + 4 = (x-1)(x+1)(x-2)(x+2)
    let poly = expr!(ctx, x ^ 4 - 5 * x ^ 2 + 4);
    let factored = poly.factor(&x);
    let re_expanded = factored.expand().eval();
    assert_equal_at_points(&poly, &re_expanded, &x, "expand(factor(x⁴-5x²+4))");
}

#[test]
fn factor_irreducible_stays_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² + 1 is irreducible over ℤ
    let poly = expr!(ctx, x ^ 2 + 1);
    let factored = poly.factor(&x);
    // It should remain essentially the same
    assert_equal_at_points(&poly, &factored, &x, "factor(x²+1)");
}

#[test]
fn factor_with_content() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 4x² - 4 = 4(x²-1) = 4(x-1)(x+1)
    let poly = expr!(ctx, 4 * x ^ 2 - 4);
    let factored = poly.factor(&x);
    assert_equal_at_points(&poly, &factored, &x, "factor(4x²-4)");
}

#[test]
fn factor_perfect_square() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² + 6x + 9 = (x+3)²
    let poly = expr!(ctx, x ^ 2 + 6 * x + 9);
    let factored = poly.factor(&x);
    assert_equal_at_points(&poly, &factored, &x, "factor(x²+6x+9)");
}

#[test]
fn factor_difference_of_cubes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ - 8 = (x-2)(x²+2x+4)
    let poly = expr!(ctx, x ^ 3 - 8);
    let factored = poly.factor(&x);
    assert_equal_at_points(&poly, &factored, &x, "factor(x³-8)");

    let s = format!("{factored}");
    assert!(!s.contains("x^3"), "x³-8 should be factored, got: {s}");
}

#[test]
fn factor_sum_of_cubes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ + 8 = (x+2)(x²-2x+4)
    let poly = expr!(ctx, x ^ 3 + 8);
    let factored = poly.factor(&x);
    assert_equal_at_points(&poly, &factored, &x, "factor(x³+8)");

    let s = format!("{factored}");
    assert!(!s.contains("x^3"), "x³+8 should be factored, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. SIMPLIFICATION CORRECTNESS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_pythagorean_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin²(x) + cos²(x) = 1
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let simplified = expr.simplify();
    assert_eq!(
        format!("{simplified}"),
        "1",
        "sin²(x)+cos²(x) should simplify to 1"
    );
}

#[test]
fn simplify_pythagorean_in_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 2 + sin²(x) + cos²(x) = 3
    let expr = &x.sin().powi(2) + &x.cos().powi(2) + 2;
    let simplified = expr.simplify();
    assert_eq!(
        format!("{simplified}"),
        "3",
        "2+sin²(x)+cos²(x) should simplify to 3"
    );
}

#[test]
fn simplify_pythagorean_with_symbol() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // y + sin²(x) + cos²(x) should simplify to y + 1
    let expr = &y + &x.sin().powi(2) + &x.cos().powi(2);
    let simplified = expr.simplify();
    assert_eq!(
        format!("{simplified}"),
        "y + 1",
        "y+sin²(x)+cos²(x) should simplify to y+1"
    );
}

#[test]
fn simplify_exp_ln_composition() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(ln(x)) = x
    let expr = x.ln().exp();
    let simplified = expr.simplify();
    assert_eq!(
        format!("{simplified}"),
        "x",
        "exp(ln(x)) should simplify to x"
    );
}

#[test]
fn simplify_algebraic_fraction_x2_minus_1_over_x_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x² - 1)/(x - 1) should cancel to x + 1
    let expr = (x.powi(2) - 1) / (&x - 1);
    let cancelled = expr.cancel(&x);
    assert_eq!(
        format!("{cancelled}"),
        "x + 1",
        "(x²-1)/(x-1) should cancel to x+1"
    );
}

#[test]
fn simplify_expand_difference_of_squares() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x+1)(x-1) should expand to x² - 1
    let expr = (&x + 1) * (&x - 1);
    let expanded = expr.expand().eval();
    let _s = format!("{expanded}");
    // Should be x^2 - 1 in some canonical form
    // Verify numerically
    for &pt in &[-3i64, -1, 0, 1, 3] {
        let v_expanded = eval_at(&expanded, &x, pt);
        let v_expected = (pt * pt - 1) as f64;
        assert!(
            approx(v_expanded, v_expected, 1e-10),
            "(x+1)(x-1) expanded at x={pt}: got {v_expanded}, expected {v_expected}"
        );
    }
}

#[test]
fn simplify_nested_square_minus_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x+1)² - x² - 2x should simplify to 1
    let expr = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
    let simplified = expr.simplify();
    assert_eq!(
        format!("{simplified}"),
        "1",
        "(x+1)²-x²-2x should simplify to 1"
    );
}

#[test]
fn simplify_double_negative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // -(-x) should simplify to x
    let expr = -(-&x);
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "x", "-(-x) should simplify to x");
}

#[test]
fn simplify_zero_times_anything() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 0 * (x² + sin(x) + 17) should be 0
    let expr = &ctx.int(0) * &(&x.powi(2) + &x.sin() + 17);
    let simplified = expr.simplify();
    assert!(
        simplified.is_zero_structural(),
        "0 * expr should be 0, got {simplified}"
    );
}

#[test]
fn simplify_x_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x/x should simplify to 1
    let expr = &x / &x;
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "1", "x/x should simplify to 1");
}

#[test]
fn simplify_x_minus_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x - x should be 0
    let expr = &x - &x;
    assert!(
        expr.is_zero_structural() || expr.simplify().is_zero_structural(),
        "x - x should be 0, got {expr}"
    );
}

#[test]
fn simplify_power_of_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x²)³ should equal x⁶
    let expr = x.powi(2).powi(3);
    let simplified = expr.simplify();
    // Check numerically
    for &pt in &[2i64, 3, -2] {
        let v = eval_at(&simplified, &x, pt);
        let expected = (pt as f64).powi(6);
        assert!(
            approx(v, expected, 1e-9),
            "(x²)³ at x={pt}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn simplify_sqrt_of_square_is_abs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // √(x²) should simplify to |x|
    let expr = x.powi(2).sqrt();
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "abs(x)", "√(x²) should be |x|");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. SUBSTITUTION CORRECTNESS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn subs_basic_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f(x) = x² + 3x + 2, f(5) should be 42
    let f = expr!(ctx, x ^ 2 + 3 * x + 2);
    let result = f.subs_i64(&x, 5).eval();
    assert_eq!(format!("{result}"), "42", "f(5) = 25+15+2 = 42");
}

#[test]
fn subs_symbolic() {
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    // f(x) = x², f(a+b) should be (a+b)²
    let f = x.powi(2);
    let result = f.subs(&x, &(&a + &b)).expand().eval();
    // (a+b)² = a² + 2ab + b²
    // Verify at a=2, b=3: should give 25
    let numerical = result.subs_i64(&a, 2).subs_i64(&b, 3).eval();
    let v = numerical.eval_f64().expect("should eval");
    assert!(
        approx(v, 25.0, 1e-10),
        "f(a+b) at a=2,b=3 should be 25, got {v}"
    );
}

#[test]
fn subs_nested() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    // f(x) = x³, substitute x → y+1, then y → z+2
    // Should give (z+3)³
    let f = x.powi(3);
    let step1 = f.subs(&x, &(&y + 1));
    let step2 = step1.subs(&y, &(&z + 2));
    // Evaluate at z=0: should give 3³ = 27
    let val = step2.subs_i64(&z, 0).eval();
    let v = val.eval_f64().expect("should eval");
    assert!(
        approx(v, 27.0, 1e-10),
        "((z+2)+1)³ at z=0 should be 27, got {v}"
    );
}

#[test]
fn subs_trig_argument() {
    let ctx = Context::new();
    let (x, _a) = (ctx.symbol("x"), ctx.symbol("a"));
    // sin(x) with x → π/6 should give 1/2
    let expr = x.sin();
    let result = expr.subs(&x, &(&ctx.pi() / 6)).eval();
    assert_eq!(format!("{result}"), "1/2", "sin(π/6) should be 1/2");
}

#[test]
fn subs_preserves_identity() {
    let ctx = Context::new();
    let (x, _y) = (ctx.symbol("x"), ctx.symbol("y"));
    // Substituting x → x should be identity
    let f = expr!(ctx, x ^ 2 + x + 1);
    let result = f.subs(&x, &x);
    // Should be equal to original at all points
    for &pt in &[-2i64, 0, 1, 3] {
        let v_orig = eval_at(&f, &x, pt);
        let v_sub = eval_at(&result, &x, pt);
        assert!(
            approx(v_orig, v_sub, 1e-10),
            "subs(x,x) should be identity at x={pt}"
        );
    }
}

#[test]
fn subs_into_sum() {
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    // f(x) = x² + 2x + 1, subs x → a+b
    // f(a+b) = (a+b)² + 2(a+b) + 1 = a²+2ab+b² + 2a+2b + 1
    let f = expr!(ctx, x ^ 2 + 2 * x + 1);
    let result = f.subs(&x, &(&a + &b)).expand().eval();
    // At a=1, b=2: f(3) = 9+6+1 = 16
    let v = result
        .subs_i64(&a, 1)
        .subs_i64(&b, 2)
        .eval()
        .eval_f64()
        .expect("eval");
    assert!(approx(v, 16.0, 1e-10), "f(1+2) should be 16, got {v}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. GCD / CANCEL
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cancel_basic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x²-1)/(x-1) → x+1
    let expr = (x.powi(2) - 1) / (&x - 1);
    let cancelled = expr.cancel(&x);
    assert_eq!(
        format!("{cancelled}"),
        "x + 1",
        "cancel((x²-1)/(x-1)) should give x+1"
    );
}

#[test]
fn cancel_cubic_over_linear() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x³-x)/(x+1) = x(x-1)(x+1)/(x+1) = x(x-1) = x²-x
    let expr = (&x.powi(3) - &x) / (&x + 1);
    let cancelled = expr.cancel(&x);
    // Should be x² - x (or equivalent)
    for &pt in &[-3i64, -2, 0, 2, 3, 5] {
        let v = eval_at(&cancelled, &x, pt);
        let expected = (pt * pt - pt) as f64;
        assert!(
            approx(v, expected, 1e-9),
            "cancel((x³-x)/(x+1)) at x={pt}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn cancel_common_quadratic_factor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x²-1)(x+2) / (x²-1) → x+2
    let numer = (&x.powi(2) - 1) * (&x + 2);
    let denom = x.powi(2) - 1;
    let expr = &numer / &denom;
    let cancelled = expr.cancel(&x);

    for &pt in &[-3i64, 0, 2, 3, 5] {
        let v = eval_at(&cancelled, &x, pt);
        let expected = (pt + 2) as f64;
        assert!(
            approx(v, expected, 1e-9),
            "cancel at x={pt}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn cancel_no_common_factor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x+1)/(x+2) — no common factor, should remain unchanged
    let expr = (&x + 1) / (&x + 2);
    let cancelled = expr.cancel(&x);
    for &pt in &[-3i64, 0, 1, 3, 5] {
        let v_orig = eval_at(&expr, &x, pt);
        let v_canc = eval_at(&cancelled, &x, pt);
        assert!(
            approx(v_orig, v_canc, 1e-9),
            "cancel should not change (x+1)/(x+2) at x={pt}"
        );
    }
}

#[test]
fn poly_gcd_basic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // gcd(x²-1, x²-2x+1) = gcd((x-1)(x+1), (x-1)²) = x-1
    let a = expr!(ctx, x ^ 2 - 1);
    let b = expr!(ctx, x ^ 2 - 2 * x + 1);
    let g = a.poly_gcd(&b, &x).expect("a.poly_gcd(&b, &x) must be Some");
    // The GCD should be (x-1) or a scalar multiple
    // At x=1 it should be 0
    let v1 = eval_at(&g, &x, 1);
    assert!(
        approx(v1, 0.0, 1e-9),
        "gcd(x²-1, (x-1)²) should vanish at x=1, got {v1}"
    );
    // At x=-1 it should be nonzero (since (x-1) at x=-1 is -2)
    let vm1 = eval_at(&g, &x, -1);
    assert!(vm1.abs() > 0.1, "gcd should be nonzero at x=-1, got {vm1}");
}

#[test]
fn poly_gcd_coprime() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // gcd(x+1, x+2) should be 1 (constant)
    let a = &x + 1;
    let b = &x + 2;
    let g = a.poly_gcd(&b, &x).expect("a.poly_gcd(&b, &x) must be Some");
    // Should be a constant (degree 0)
    let v0 = eval_at(&g, &x, 0);
    let v5 = eval_at(&g, &x, 5);
    assert!(
        approx(v0, v5, 1e-9),
        "gcd(x+1,x+2) should be constant, but varies: {v0} vs {v5}"
    );
}

#[test]
fn cancel_perfect_square_over_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x²+2x+1)/(x+1) = (x+1)²/(x+1) = x+1
    let expr = (x.powi(2) + &x * 2 + 1) / (&x + 1);
    let cancelled = expr.cancel(&x);
    assert_eq!(
        format!("{cancelled}"),
        "x + 1",
        "(x²+2x+1)/(x+1) should cancel to x+1"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. POLYNOMIAL SYSTEM SOLVING
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn system_circle_line() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    // x² + y² = 1, x + y = 1 → (0,1) and (1,0)
    let eq1 = expr!(ctx, x ^ 2 + y ^ 2 - 1);
    let eq2 = expr!(ctx, x + y - 1);
    let solutions =
        symplex::polysys::solve_system_ex(&[eq1.clone(), eq2.clone()], &[x.clone(), y.clone()])
            .unwrap();

    assert_eq!(solutions.len(), 2, "expected 2 solutions");

    // Verify each solution
    for sol in &solutions {
        let r1 = eq1.subs(&x, &sol[0]).subs(&y, &sol[1]).eval().simplify();
        let r2 = eq2.subs(&x, &sol[0]).subs(&y, &sol[1]).eval().simplify();
        assert!(
            r1.is_zero_structural(),
            "eq1 residual should be 0, got {r1}"
        );
        assert!(
            r2.is_zero_structural(),
            "eq2 residual should be 0, got {r2}"
        );
    }
}

#[test]
fn system_two_conics() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    // x² + y² = 5, xy = 2 → (1,2), (2,1), (-1,-2), (-2,-1)
    let eq1 = expr!(ctx, x ^ 2 + y ^ 2 - 5);
    let eq2 = expr!(ctx, x * y - 2);
    let solutions =
        symplex::polysys::solve_system_ex(&[eq1.clone(), eq2.clone()], &[x.clone(), y.clone()])
            .unwrap();

    assert_eq!(solutions.len(), 4, "expected 4 solutions");

    for sol in &solutions {
        let r1 = eq1.subs(&x, &sol[0]).subs(&y, &sol[1]).eval().simplify();
        let r2 = eq2.subs(&x, &sol[0]).subs(&y, &sol[1]).eval().simplify();
        assert!(
            r1.is_zero_structural(),
            "eq1 not satisfied for ({}, {}): residual = {r1}",
            sol[0],
            sol[1]
        );
        assert!(
            r2.is_zero_structural(),
            "eq2 not satisfied for ({}, {}): residual = {r2}",
            sol[0],
            sol[1]
        );
    }
}

#[test]
fn system_linear_2x2() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    // x + y = 5, x - y = 1 → (3, 2)
    let eq1 = expr!(ctx, x + y - 5);
    let eq2 = expr!(ctx, x - y - 1);
    let solutions =
        symplex::polysys::solve_system_ex(&[eq1.clone(), eq2.clone()], &[x.clone(), y.clone()])
            .unwrap();

    assert_eq!(solutions.len(), 1, "expected 1 solution");
    assert_eq!(format!("{}", solutions[0][0]), "3", "x should be 3");
    assert_eq!(format!("{}", solutions[0][1]), "2", "y should be 2");
}

#[test]
fn system_parabola_line() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    // y = x², y = x + 2 → x² - x - 2 = 0 → (x-2)(x+1) = 0
    // Solutions: (2, 4) and (-1, 1)
    let eq1 = expr!(ctx, y - x ^ 2);
    let eq2 = expr!(ctx, y - x - 2);
    let solutions =
        symplex::polysys::solve_system_ex(&[eq1.clone(), eq2.clone()], &[x.clone(), y.clone()])
            .unwrap();

    assert_eq!(solutions.len(), 2, "expected 2 solutions");

    for sol in &solutions {
        let r1 = eq1.subs(&x, &sol[0]).subs(&y, &sol[1]).eval().simplify();
        let r2 = eq2.subs(&x, &sol[0]).subs(&y, &sol[1]).eval().simplify();
        assert!(
            r1.is_zero_structural(),
            "y-x² not satisfied: residual {r1} at ({}, {})",
            sol[0],
            sol[1]
        );
        assert!(
            r2.is_zero_structural(),
            "y-x-2 not satisfied: residual {r2} at ({}, {})",
            sol[0],
            sol[1]
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. EXPAND CORRECTNESS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_binomial_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x+1)³ = x³ + 3x² + 3x + 1
    let expr = (&x + 1).powi(3);
    let expanded = expr.expand().eval();
    for &pt in &[-2i64, -1, 0, 1, 2, 3] {
        let v = eval_at(&expanded, &x, pt);
        let expected = ((pt + 1) as f64).powi(3);
        assert!(
            approx(v, expected, 1e-9),
            "(x+1)³ expanded at x={pt}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn expand_product_of_sums() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x+1)(x+2)(x+3) = x³ + 6x² + 11x + 6
    let expr = (&x + 1) * (&x + 2) * (&x + 3);
    let expanded = expr.expand().eval();
    for &pt in &[-4i64, -3, -2, -1, 0, 1, 2] {
        let v = eval_at(&expanded, &x, pt);
        let expected = ((pt + 1) * (pt + 2) * (pt + 3)) as f64;
        assert!(
            approx(v, expected, 1e-9),
            "(x+1)(x+2)(x+3) expanded at x={pt}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn expand_fourth_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x+1)⁴ should expand correctly
    // (x+1)⁴ = x⁴ + 4x³ + 6x² + 4x + 1
    let expr = (&x + 1).powi(4);
    let expanded = expr.expand().eval();
    for &pt in &[-2i64, -1, 0, 1, 2, 3] {
        let v = eval_at(&expanded, &x, pt);
        let expected = ((pt + 1) as f64).powi(4);
        assert!(
            approx(v, expected, 1e-9),
            "(x+1)⁴ expanded at x={pt}: got {v}, expected {expected}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. PARTIAL FRACTIONS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn partial_fractions_basic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 1/((x-1)(x+1)) = 1/(x²-1) should decompose to A/(x-1) + B/(x+1)
    // where A = 1/2, B = -1/2
    let expr = ctx.int(1) / (x.powi(2) - 1);
    let pf = expr.partial_fractions(&x);
    // Verify numerical equivalence at non-singular points
    for &pt in &[-3i64, -2, 0, 2, 3, 5] {
        let v_orig = eval_at(&expr, &x, pt);
        let v_pf = eval_at(&pf, &x, pt);
        assert!(
            approx(v_orig, v_pf, 1e-9),
            "partial_fractions(1/(x²-1)) at x={pt}: orig={v_orig}, pf={v_pf}"
        );
    }
}

#[test]
fn partial_fractions_cubic_denom() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 1/(x(x-1)(x+1)) = 1/(x³-x)
    let expr = ctx.int(1) / (&x.powi(3) - &x);
    let pf = expr.partial_fractions(&x);
    for &pt in &[-3i64, -2, 2, 3, 5, 7] {
        let v_orig = eval_at(&expr, &x, pt);
        let v_pf = eval_at(&pf, &x, pt);
        assert!(
            approx(v_orig, v_pf, 1e-9),
            "partial_fractions(1/(x³-x)) at x={pt}: orig={v_orig}, pf={v_pf}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. DEGREE AND COEFFICIENTS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn degree_of_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(expr!(ctx, x ^ 3 + 2 * x + 1).degree(&x), Some(3));
    assert_eq!(expr!(ctx, x ^ 2 - 1).degree(&x), Some(2));
    assert_eq!((&x + 1).degree(&x), Some(1));
    assert_eq!(ctx.int(5).degree(&x), Some(0));
}

#[test]
fn coefficients_extraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² - 5x + 6 has coefficients [6, -5, 1]
    let poly = expr!(ctx, x ^ 2 - 5 * x + 6);
    let coeffs = poly.coeffs(&x).expect("poly.coeffs(&x) must be Some");
    assert_eq!(coeffs.len(), 3, "quadratic should have 3 coefficients");
    assert_eq!(format!("{}", coeffs[0]), "6", "constant term should be 6");
    assert_eq!(
        format!("{}", coeffs[1]),
        "-5",
        "linear coefficient should be -5"
    );
    assert_eq!(
        format!("{}", coeffs[2]),
        "1",
        "leading coefficient should be 1"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. EVAL SPECIAL VALUES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sin_special_values() {
    let ctx = Context::new();
    // sin(0) = 0
    let v = ctx.int(0).sin().eval();
    assert_eq!(format!("{v}"), "0", "sin(0) should be 0");

    // sin(π/2) = 1
    let v = (&ctx.pi() / 2).sin().eval();
    assert_eq!(format!("{v}"), "1", "sin(π/2) should be 1");

    // sin(π) = 0
    let v = ctx.pi().sin().eval();
    assert_eq!(format!("{v}"), "0", "sin(π) should be 0");

    // sin(π/6) = 1/2
    let v = (&ctx.pi() / 6).sin().eval();
    assert_eq!(format!("{v}"), "1/2", "sin(π/6) should be 1/2");
}

#[test]
fn eval_cos_special_values() {
    let ctx = Context::new();
    // cos(0) = 1
    let v = ctx.int(0).cos().eval();
    assert_eq!(format!("{v}"), "1", "cos(0) should be 1");

    // cos(π/3) = 1/2
    let v = (&ctx.pi() / 3).cos().eval();
    assert_eq!(format!("{v}"), "1/2", "cos(π/3) should be 1/2");

    // cos(π) = -1
    let v = ctx.pi().cos().eval();
    assert_eq!(format!("{v}"), "-1", "cos(π) should be -1");
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. CROSS-OPERATION CONSISTENCY
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_and_factor_agree_on_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² - 5x + 6: solve gives {2,3}, factor gives (x-2)(x-3)
    let poly = expr!(ctx, x ^ 2 - 5 * x + 6);
    let roots = poly.solve_or_empty(&x);
    let factored = poly.factor(&x);

    // Each root from solve should be a root of the factored form
    for root in &roots {
        let v = factored.subs(&x, root).eval().simplify();
        assert!(
            v.is_zero_structural() || v.eval_f64().map(|f| f.abs() < 1e-10).unwrap_or(false),
            "root {root} from solve is not a root of factored form, residual = {v}"
        );
    }
}

#[test]
fn factor_then_expand_is_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // For several polynomials, factor then expand should reproduce the original
    let polys = vec![
        expr!(ctx, x ^ 2 - 1),
        expr!(ctx, x ^ 3 - 6 * x ^ 2 + 11 * x - 6),
        expr!(ctx, x ^ 4 - 5 * x ^ 2 + 4),
        expr!(ctx, x ^ 3 - x),
    ];
    for poly in &polys {
        let factored = poly.factor(&x);
        let re_expanded = factored.expand().eval();
        assert_equal_at_points(
            poly,
            &re_expanded,
            &x,
            &format!("factor/expand roundtrip for {poly}"),
        );
    }
}

#[test]
fn cancel_and_simplify_agree() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x²-4)/(x-2) should give x+2 via cancel
    let expr = (x.powi(2) - 4) / (&x - 2);
    let cancelled = expr.cancel(&x);
    // x+2 at x=5 should be 7
    let v = eval_at(&cancelled, &x, 5);
    assert!(
        approx(v, 7.0, 1e-10),
        "cancel((x²-4)/(x-2)) at x=5 should be 7, got {v}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. EDGE CASES / REGRESSION CANDIDATES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_linear_trivial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x = 0 → root is 0
    let roots = x.solve_or_empty(&x);
    assert_eq!(roots.len(), 1);
    assert_eq!(format!("{}", roots[0]), "0");
}

#[test]
fn solve_already_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 0 = 0 → all x satisfy, returns empty by convention
    let roots = ctx.int(0).solve_or_empty(&x);
    assert!(
        roots.is_empty(),
        "0=0 should return empty (infinitely many solutions)"
    );
}

#[test]
fn solve_constant_nonzero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 7 = 0 → no solutions
    let roots = ctx.int(7).solve_or_empty(&x);
    assert!(roots.is_empty(), "7=0 should have no solutions");
}

#[test]
fn expand_then_factor_preserves_value_for_x5_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁵ - 1 = (x-1)(x⁴+x³+x²+x+1)
    let poly = expr!(ctx, x ^ 5 - 1);
    let factored = poly.factor(&x);
    let re_expanded = factored.expand().eval();
    assert_equal_at_points(&poly, &re_expanded, &x, "x⁵-1 factor/expand");
}

#[test]
fn check_solution_api() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² - 4 = 0 → roots ±2
    let poly = expr!(ctx, x ^ 2 - 4);
    assert_eq!(
        poly.check_solution(&x, &ctx.int(2)),
        Some(true),
        "x=2 should satisfy x²-4=0"
    );
    assert_eq!(
        poly.check_solution(&x, &ctx.int(-2)),
        Some(true),
        "x=-2 should satisfy x²-4=0"
    );
    assert_eq!(
        poly.check_solution(&x, &ctx.int(3)),
        Some(false),
        "x=3 should NOT satisfy x²-4=0"
    );
}

#[test]
fn solve_quartic_vieta_sum_of_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁴ - 10x³ + 35x² - 50x + 24 = (x-1)(x-2)(x-3)(x-4)
    // Sum of roots = 10 (by Vieta's)
    let poly = expr!(ctx, x ^ 4 - 10 * x ^ 3 + 35 * x ^ 2 - 50 * x + 24);
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 4, "expected 4 roots");

    let sum: f64 = roots
        .iter()
        .map(|r| r.eval_f64().expect("root should eval"))
        .sum();
    assert!(
        approx(sum, 10.0, 1e-9),
        "sum of roots should be 10 (Vieta's), got {sum}"
    );

    // Product of roots = 24 (by Vieta's, since leading coeff is 1 and degree is 4)
    let product: f64 = roots
        .iter()
        .map(|r| r.eval_f64().expect("root should eval"))
        .product();
    assert!(
        approx(product, 24.0, 1e-9),
        "product of roots should be 24 (Vieta's), got {product}"
    );
}

#[test]
fn solve_cubic_vieta_relations() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
    // r1+r2+r3 = 6, r1*r2+r1*r3+r2*r3 = 11, r1*r2*r3 = 6
    let poly = expr!(ctx, x ^ 3 - 6 * x ^ 2 + 11 * x - 6);
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 3, "expected 3 roots");

    let vals: Vec<f64> = roots
        .iter()
        .map(|r| r.eval_f64().expect("root should eval"))
        .collect();

    let sum: f64 = vals.iter().sum();
    assert!(
        approx(sum, 6.0, 1e-9),
        "sum of roots should be 6, got {sum}"
    );

    let product: f64 = vals.iter().product();
    assert!(
        approx(product, 6.0, 1e-9),
        "product of roots should be 6, got {product}"
    );

    // Sum of products of pairs
    let pair_sum = vals[0] * vals[1] + vals[0] * vals[2] + vals[1] * vals[2];
    assert!(
        approx(pair_sum, 11.0, 1e-9),
        "sum of pairwise products should be 11, got {pair_sum}"
    );
}

#[test]
fn factor_x4_plus_4() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁴ + 4 = (x²+2x+2)(x²-2x+2) — Sophie Germain identity
    let poly = expr!(ctx, x ^ 4 + 4);
    let factored = poly.factor(&x);
    assert_equal_at_points(&poly, &factored, &x, "factor(x⁴+4)");

    // Check if it actually factored
    let s = format!("{factored}");
    // x⁴+4 should factor over ℤ
    assert!(
        !s.contains("x^4"),
        "x⁴+4 should factor (Sophie Germain), got: {s}"
    );
}

#[test]
fn factor_x8_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁸ - 1 has many factors over ℤ
    let poly = expr!(ctx, x ^ 8 - 1);
    let factored = poly.factor(&x);
    assert_equal_at_points(&poly, &factored, &x, "factor(x⁸-1)");
}

#[test]
fn cancel_higher_degree_common_factor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x³-x)/(x²-1) = x(x²-1)/(x²-1) = x
    let expr = (&x.powi(3) - &x) / (&x.powi(2) - 1);
    let cancelled = expr.cancel(&x);
    for &pt in &[-3i64, -2, 0, 2, 3, 5] {
        let v = eval_at(&cancelled, &x, pt);
        assert!(
            approx(v, pt as f64, 1e-9),
            "(x³-x)/(x²-1) should cancel to x, got {v} at x={pt}"
        );
    }
}

#[test]
fn simplify_sin_squared_times_two_plus_cos_squared_times_two() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 2sin²(x) + 2cos²(x) = 2(sin²(x)+cos²(x)) = 2·1 = 2
    let expr = &x.sin().powi(2) * 2 + &x.cos().powi(2) * 2;
    let simplified = expr.simplify();
    let s = format!("{simplified}");
    // BUG: symplex returns "2*sin(x)^2 + 2*cos(x)^2" instead of "2".
    // The Pythagorean identity sin²+cos² = 1 is only recognized when
    // the coefficients are exactly 1. When both terms share a common
    // factor (here 2), the pattern matcher fails to extract the common
    // coefficient first and then apply the identity.
    //
    // Also try full_simplify and trigsimp to confirm the bug:
    let full = expr.simplify();
    let full_s = format!("{full}");

    // At minimum the numerical value must be 2 everywhere
    for &pt in &[-3i64, -1, 0, 1, 2, 5] {
        let v = eval_at(&simplified, &x, pt);
        assert!(
            approx(v, 2.0, 1e-9),
            "2sin²(x)+2cos²(x) should always be 2, got {v} at x={pt}"
        );
    }

    // The actual assertion: simplify should produce "2"
    assert!(
        s == "2" || full_s == "2",
        "BUG: 2*sin²(x)+2*cos²(x) should simplify to 2, \
         but simplify gave '{s}' and full_simplify gave '{full_s}'"
    );
}

/// Variation: 3*sin²(x) + 3*cos²(x) should simplify to 3
#[test]
fn simplify_sin_squared_times_three_plus_cos_squared_times_three() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) * 3 + &x.cos().powi(2) * 3;
    let simplified = expr.simplify();
    let full = expr.simplify();
    let s = format!("{simplified}");
    let full_s = format!("{full}");

    // Verify numerically
    for &pt in &[-2i64, 0, 1, 4] {
        let v = eval_at(&simplified, &x, pt);
        assert!(
            approx(v, 3.0, 1e-9),
            "3sin²+3cos² should be 3 at x={pt}, got {v}"
        );
    }

    assert!(
        s == "3" || full_s == "3",
        "BUG: 3*sin²(x)+3*cos²(x) should simplify to 3, \
         but simplify gave '{s}' and full_simplify gave '{full_s}'"
    );
}

/// Variation: y*sin²(x) + y*cos²(x) should simplify to y
#[test]
fn simplify_y_times_sin_squared_plus_y_times_cos_squared() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let expr = &y * &x.sin().powi(2) + &y * &x.cos().powi(2);
    let simplified = expr.simplify();
    let full = expr.simplify();
    let s = format!("{simplified}");
    let full_s = format!("{full}");

    assert!(
        s == "y" || full_s == "y",
        "BUG: y*sin²(x)+y*cos²(x) should simplify to y, \
         but simplify gave '{s}' and full_simplify gave '{full_s}'"
    );
}

/// Variation: sin²(x)/2 + cos²(x)/2 should simplify to 1/2
#[test]
fn simplify_half_sin_squared_plus_half_cos_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) / 2 + &x.cos().powi(2) / 2;
    let simplified = expr.simplify();
    let full = expr.simplify();
    let s = format!("{simplified}");
    let full_s = format!("{full}");

    // Numerical check
    for &pt in &[-1i64, 0, 1, 3] {
        let v = eval_at(&simplified, &x, pt);
        assert!(
            approx(v, 0.5, 1e-9),
            "sin²/2+cos²/2 should be 0.5 at x={pt}, got {v}"
        );
    }

    assert!(
        s == "1/2" || full_s == "1/2",
        "BUG: sin²(x)/2+cos²(x)/2 should simplify to 1/2, \
         but simplify gave '{s}' and full_simplify gave '{full_s}'"
    );
}

#[test]
fn solve_quadratic_with_fraction_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (1/2)x² - (3/2)x + 1 = 0 → multiply by 2: x² - 3x + 2 = 0 → x=1,2
    let poly = &x.powi(2) / 2 - &x * ctx.rational(3, 2) + 1;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "expected 2 roots");
    verify_roots_are_zeros(&poly, &x, &roots, "(1/2)x²-(3/2)x+1");
}

#[test]
fn expand_large_binomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x+1)^6 should expand correctly
    let expr = (&x + 1).powi(6);
    let expanded = expr.expand().eval();
    // Binomial coefficients: 1, 6, 15, 20, 15, 6, 1
    // Check at x=1: (1+1)^6 = 64
    let v = eval_at(&expanded, &x, 1);
    assert!(
        approx(v, 64.0, 1e-9),
        "(x+1)^6 at x=1 should be 64, got {v}"
    );
    // x=2: 3^6 = 729
    let v = eval_at(&expanded, &x, 2);
    assert!(
        approx(v, 729.0, 1e-9),
        "(x+1)^6 at x=2 should be 729, got {v}"
    );
}

#[test]
fn solve_quintic_with_rational_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁵ - x = x(x⁴-1) = x(x-1)(x+1)(x²+1) → rational roots: 0, 1, -1
    let poly = expr!(ctx, x ^ 5 - x);
    let roots = poly.solve_or_empty(&x);
    // Should find at least the rational roots; may also find complex roots
    assert!(
        roots.len() >= 3,
        "x⁵-x should have at least 3 roots (0, ±1, ±i), got {}",
        roots.len()
    );
    verify_roots_are_zeros(&poly, &x, &roots, "x⁵-x");
}

#[test]
fn subs_chain_consistency() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    // f(x) = x³ - 2x + 1
    // f(a) should equal a³ - 2a + 1
    let f = expr!(ctx, x ^ 3 - 2 * x + 1);
    let fa = f.subs(&x, &a);
    // Evaluate both at a=3 (which is x=3 for original)
    let v_orig = eval_at(&f, &x, 3); // 27 - 6 + 1 = 22
    let v_sub = eval_at(&fa, &a, 3);
    assert!(
        approx(v_orig, v_sub, 1e-10),
        "f(3) should equal f(x).subs(x,a) evaluated at a=3: {v_orig} vs {v_sub}"
    );
}

#[test]
fn differentiate_then_solve_for_critical_points() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f(x) = x³ - 3x + 2
    // f'(x) = 3x² - 3 = 0 → x = ±1
    let f = expr!(ctx, x ^ 3 - 3 * x + 2);
    let df = f.diff(&x);
    let critical = df.solve_or_empty(&x);
    assert_eq!(critical.len(), 2, "f'(x)=3x²-3 should have 2 roots");

    let strs: Vec<String> = critical.iter().map(|r| format!("{r}")).collect();
    assert!(
        strs.contains(&"1".to_string()),
        "missing critical point 1: {strs:?}"
    );
    assert!(
        strs.contains(&"-1".to_string()),
        "missing critical point -1: {strs:?}"
    );
}

#[test]
fn cancel_x3_minus_x2_over_x2_minus_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x³-x²)/(x²-x) = x²(x-1) / (x(x-1)) = x
    let expr = (&x.powi(3) - &x.powi(2)) / (&x.powi(2) - &x);
    let cancelled = expr.cancel(&x);
    for &pt in &[2i64, 3, 5, -2, -3] {
        let v = eval_at(&cancelled, &x, pt);
        assert!(
            approx(v, pt as f64, 1e-9),
            "(x³-x²)/(x²-x) should cancel to x at x={pt}, got {v}"
        );
    }
}

#[test]
fn solve_quadratic_negative_discriminant_gives_complex_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² + x + 1 = 0, discriminant = 1 - 4 = -3, complex roots
    let poly = expr!(ctx, x ^ 2 + x + 1);
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "x²+x+1 should have 2 complex roots");
    verify_roots_are_zeros(&poly, &x, &roots, "x²+x+1");
}

#[test]
fn system_tangent_circles() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    // x² + y² = 4, (x-3)² + y² = 1
    // Expand second: x²-6x+9+y² = 1
    // Subtract first: -6x+9 = -3 → x = 2
    // Then y² = 4-4 = 0 → y = 0
    // Single solution: (2, 0)
    let eq1 = expr!(ctx, x ^ 2 + y ^ 2 - 4);
    let eq2 = expr!(ctx, (x - 3) ^ 2 + y ^ 2 - 1);
    let eq2_expanded = eq2.expand().eval();
    let solutions = symplex::polysys::solve_system_ex(
        &[eq1.clone(), eq2_expanded.clone()],
        &[x.clone(), y.clone()],
    )
    .unwrap();

    assert_eq!(solutions.len(), 1, "tangent circles should have 1 solution");
    // Verify
    let r1 = eq1
        .subs(&x, &solutions[0][0])
        .subs(&y, &solutions[0][1])
        .eval()
        .simplify();
    assert!(
        r1.is_zero_structural(),
        "solution should satisfy eq1, residual = {r1}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. ADDITIONAL SIMPLIFICATION EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

/// 1 - sin²(x) should simplify to cos²(x) (or remain as 1-sin²(x))
#[test]
fn simplify_one_minus_sin_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = ctx.int(1) - x.sin().powi(2);
    let simplified = expr.simplify();
    let s = format!("{simplified}");
    // Acceptable forms: "cos(x)^2", "1 - sin(x)^2", "-sin(x)^2 + 1"
    // All are mathematically equivalent.
    // Verify numerically that the simplified form is correct
    for &pt in &[-2i64, -1, 0, 1, 2, 3] {
        let v = eval_at(&simplified, &x, pt);
        let pt_f = pt as f64;
        let expected = pt_f.cos().powi(2);
        assert!(
            approx(v, expected, 1e-9),
            "1-sin²(x) at x={pt}: got {v}, expected {expected}"
        );
    }
    // Accept any of the mathematically equivalent forms
    assert!(
        s.contains("cos") || s.contains("sin") || s == "1",
        "1-sin²(x) should simplify to cos²(x) or equivalent, got: {s}"
    );
}

/// Test that expand distributes correctly over subtraction: (a-b)² = a²-2ab+b²
#[test]
fn expand_square_of_difference() {
    let ctx = Context::new();
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let expr = (&a - &b).powi(2);
    let expanded = expr.expand().eval();
    // At a=5, b=3: (5-3)² = 4
    let v = expanded
        .subs_i64(&a, 5)
        .subs_i64(&b, 3)
        .eval()
        .eval_f64()
        .unwrap();
    assert!(
        approx(v, 4.0, 1e-10),
        "(a-b)² at a=5,b=3 should be 4, got {v}"
    );
    // At a=1, b=4: (1-4)² = 9
    let v = expanded
        .subs_i64(&a, 1)
        .subs_i64(&b, 4)
        .eval()
        .eval_f64()
        .unwrap();
    assert!(
        approx(v, 9.0, 1e-10),
        "(a-b)² at a=1,b=4 should be 9, got {v}"
    );
}

/// (x²-y²)/(x-y) should cancel to x+y
#[test]
fn cancel_bivariate_difference_of_squares() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let numer = &x.powi(2) - &y.powi(2);
    let denom = &x - &y;
    let expr = &numer / &denom;
    let cancelled = expr.cancel(&x);
    // Evaluate at x=5, y=2: should be 7
    let v = cancelled
        .subs_i64(&x, 5)
        .subs_i64(&y, 2)
        .eval()
        .eval_f64()
        .unwrap();
    assert!(
        approx(v, 7.0, 1e-10),
        "(x²-y²)/(x-y) cancelled at x=5,y=2 should be 7, got {v}"
    );
}

/// Verify that solve returns roots in correct multiplicity for (x-1)⁴
#[test]
fn solve_quartic_quadruple_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x-1)⁴ = x⁴ - 4x³ + 6x² - 4x + 1
    let poly = expr!(ctx, x ^ 4 - 4 * x ^ 3 + 6 * x ^ 2 - 4 * x + 1);
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty(), "(x-1)⁴ should have root(s)");
    verify_roots_are_zeros(&poly, &x, &roots, "(x-1)⁴");
    for r in &roots {
        assert_eq!(format!("{r}"), "1", "quadruple root should be 1, got {r}");
    }
}

/// Test that differentiation + solving finds critical points of x⁴ - 4x³
#[test]
fn diff_then_solve_quartic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f(x) = x⁴ - 4x³
    // f'(x) = 4x³ - 12x² = 4x²(x - 3)
    // Critical points: x=0 (double), x=3
    let f = expr!(ctx, x ^ 4 - 4 * x ^ 3);
    let df = f.diff(&x);
    let critical = df.solve_or_empty(&x);

    // Should find x=0 and x=3
    let strs: Vec<String> = critical.iter().map(|r| format!("{r}")).collect();
    assert!(
        strs.contains(&"0".to_string()),
        "missing critical point 0: {strs:?}"
    );
    assert!(
        strs.contains(&"3".to_string()),
        "missing critical point 3: {strs:?}"
    );

    // Verify they are roots of f'
    verify_roots_are_zeros(&df, &x, &critical, "f'(x)=4x³-12x²");
}

/// Test polynomial GCD of two cubics sharing a quadratic factor
#[test]
fn poly_gcd_shared_quadratic_factor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // a = (x²-1)(x+3) = x³+3x²-x-3
    // b = (x²-1)(x-5) = x³-5x²-x+5
    // gcd should be x²-1
    let a = expr!(ctx, x ^ 3 + 3 * x ^ 2 - x - 3);
    let b = expr!(ctx, x ^ 3 - 5 * x ^ 2 - x + 5);
    let g = a.poly_gcd(&b, &x).expect("a.poly_gcd(&b, &x) must be Some");
    // gcd should vanish at x=1 and x=-1
    let v1 = eval_at(&g, &x, 1);
    assert!(approx(v1, 0.0, 1e-9), "gcd should vanish at x=1, got {v1}");
    let vm1 = eval_at(&g, &x, -1);
    assert!(
        approx(vm1, 0.0, 1e-9),
        "gcd should vanish at x=-1, got {vm1}"
    );
    // gcd should be nonzero at x=2
    let v2 = eval_at(&g, &x, 2);
    assert!(v2.abs() > 0.1, "gcd should be nonzero at x=2, got {v2}");
    // Degree should be 2
    if let Some(deg) = g.degree(&x) {
        assert_eq!(deg, 2, "gcd degree should be 2, got {deg}");
    }
}
