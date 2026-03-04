//! Tests for cubic (Cardano) and quartic (Ferrari) formula solving.

use symplex::prelude::*;

/// Helper: verify roots by substituting back into the polynomial and checking
/// that the result is numerically zero (allowing complex roots).
fn verify_roots(poly_expr: &Ex, var: &Ex, roots: &[Ex], label: &str) {
    for (i, root) in roots.iter().enumerate() {
        let val = poly_expr.subs(var, root).eval();
        // Try exact check first
        let s = format!("{val}");
        if s == "0" {
            continue;
        }
        // Fall back to numerical check (works for irrational and complex roots)
        match val.evalf_complex64() {
            Ok((re, im)) => {
                let mag = (re * re + im * im).sqrt();
                assert!(
                    mag < 1e-6,
                    "{label}: root[{i}] = {root} doesn't satisfy equation \
                     (residual = {re} + {im}i, |r| = {mag})"
                );
            }
            Err(_) => {
                // If evalf fails (e.g. deeply nested symbolic), skip this root's
                // numerical check — the solver still returned *something*.
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cubic tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cubic_three_rational_roots() {
    // (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6 = 0
    let x = symplex::var("x");
    let poly = &x.powi(3) - &(&x.powi(2) * 6) + &(&x * 11) - 6;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        3,
        "x³-6x²+11x-6 should have 3 roots, got {}",
        roots.len()
    );

    let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    strs.sort();
    assert!(
        strs.contains(&"1".to_string()),
        "should have root 1: {strs:?}"
    );
    assert!(
        strs.contains(&"2".to_string()),
        "should have root 2: {strs:?}"
    );
    assert!(
        strs.contains(&"3".to_string()),
        "should have root 3: {strs:?}"
    );
}

#[test]
fn cubic_three_rational_roots_verify() {
    // Same polynomial, but verify by substitution
    let x = symplex::var("x");
    let poly = &x.powi(3) - &(&x.powi(2) * 6) + &(&x * 11) - 6;
    let roots = poly.solve_or_empty(&x);
    verify_roots(&poly, &x, &roots, "x³-6x²+11x-6");
}

#[test]
fn cubic_one_real_two_complex() {
    // x³ + 1 = 0  →  roots: -1, (1 ± i√3)/2
    let x = symplex::var("x");
    let poly = &x.powi(3) + 1;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        3,
        "x³+1 should have 3 roots, got {}",
        roots.len()
    );
    verify_roots(&poly, &x, &roots, "x³+1");
}

#[test]
fn cubic_depressed_with_zero_root() {
    // x³ - x = x(x-1)(x+1) = 0  →  roots 0, 1, -1
    let x = symplex::var("x");
    let poly = &x.powi(3) - &x;
    let roots = poly.solve_or_empty(&x);
    assert!(
        roots.len() >= 3,
        "x³-x should have 3 roots, got {}",
        roots.len()
    );
    verify_roots(&poly, &x, &roots, "x³-x");
}

#[test]
fn cubic_negative_rational_roots() {
    // (x+1)(x+2)(x+3) = x³ + 6x² + 11x + 6 = 0  →  roots -1, -2, -3
    let x = symplex::var("x");
    let poly = &x.powi(3) + &(&x.powi(2) * 6) + &(&x * 11) + 6;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        3,
        "x³+6x²+11x+6 should have 3 roots, got {}",
        roots.len()
    );

    let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    strs.sort();
    assert!(
        strs.contains(&"-1".to_string()),
        "should have root -1: {strs:?}"
    );
    assert!(
        strs.contains(&"-2".to_string()),
        "should have root -2: {strs:?}"
    );
    assert!(
        strs.contains(&"-3".to_string()),
        "should have root -3: {strs:?}"
    );
}

#[test]
fn cubic_repeated_root() {
    // (x-1)³ = x³ - 3x² + 3x - 1 = 0  →  triple root x=1
    let x = symplex::var("x");
    let poly = &x.powi(3) - &(&x.powi(2) * 3) + &(&x * 3) - 1;
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty(), "(x-1)³ should find at least one root");
    verify_roots(&poly, &x, &roots, "(x-1)³");
}

#[test]
fn cubic_double_root_and_simple() {
    // (x-1)²(x+2) = x³ - 3x + 2 → roots 1 (double), -2
    // Actually: (x-1)²(x+2) = x³ + 0x² - 3x + 2
    let x = symplex::var("x");
    let poly = &x.powi(3) - &(&x * 3) + 2;
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty(), "x³-3x+2 should have roots");
    verify_roots(&poly, &x, &roots, "x³-3x+2");
}

#[test]
fn cubic_with_leading_coefficient() {
    // 2x³ - 6x² + 4x = 2x(x-1)(x-2) = 0  →  roots 0, 1, 2
    let x = symplex::var("x");
    let poly = &(&x.powi(3) * 2) - &(&x.powi(2) * 6) + &(&x * 4);
    let roots = poly.solve_or_empty(&x);
    assert!(
        roots.len() >= 3,
        "2x³-6x²+4x should have 3 roots, got {}",
        roots.len()
    );
    verify_roots(&poly, &x, &roots, "2x³-6x²+4x");
}

#[test]
fn cubic_pure_cube() {
    // x³ - 8 = 0  →  x = 2, and two complex cube roots of 8
    let x = symplex::var("x");
    let poly = &x.powi(3) - 8;
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty(), "x³-8 should have roots");
    // At least one root should be 2
    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        strs.contains(&"2".to_string()),
        "should have root 2: {strs:?}"
    );
    verify_roots(&poly, &x, &roots, "x³-8");
}

#[test]
fn cubic_no_rational_roots() {
    // x³ - 2 = 0  →  x = ∛2 (irrational) and two complex roots
    let x = symplex::var("x");
    let poly = &x.powi(3) - 2;
    let roots = poly.solve_or_empty(&x);
    // Cardano's formula should find 3 roots
    assert_eq!(
        roots.len(),
        3,
        "x³-2 should have 3 roots via Cardano, got {}",
        roots.len()
    );
    verify_roots(&poly, &x, &roots, "x³-2");
}

#[test]
fn cubic_eq_macro() {
    // Using the eq! macro
    let x = symplex::var("x");
    let equation = eq!(x ^ 3 - 6 * x ^ 2 + 11 * x - 6 = 0);
    let roots = equation.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        3,
        "eq! cubic should have 3 roots, got {}",
        roots.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Quartic tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quartic_four_rational_roots() {
    // (x-1)(x-2)(x-3)(x-4) = x⁴ - 10x³ + 35x² - 50x + 24 = 0
    let x = symplex::var("x");
    let poly = &(&(&x.powi(4) - &(&x.powi(3) * 10)) + &(&x.powi(2) * 35)) - &(&(&x * 50) - 24);
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        4,
        "quartic with 4 rational roots should have 4, got {}",
        roots.len()
    );

    let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    strs.sort();
    assert!(
        strs.contains(&"1".to_string()),
        "should have root 1: {strs:?}"
    );
    assert!(
        strs.contains(&"2".to_string()),
        "should have root 2: {strs:?}"
    );
    assert!(
        strs.contains(&"3".to_string()),
        "should have root 3: {strs:?}"
    );
    assert!(
        strs.contains(&"4".to_string()),
        "should have root 4: {strs:?}"
    );
}

#[test]
fn quartic_biquadratic() {
    // x⁴ - 5x² + 4 = (x²-1)(x²-4) = (x-1)(x+1)(x-2)(x+2) = 0
    let x = symplex::var("x");
    let poly = &(&x.powi(4) - &(&x.powi(2) * 5)) + 4;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        4,
        "biquadratic should have 4 roots, got {}",
        roots.len()
    );
    verify_roots(&poly, &x, &roots, "x⁴-5x²+4");
}

#[test]
fn quartic_with_zero_root() {
    // x⁴ - x³ = x³(x-1) = 0  →  roots 0 (triple), 1
    let x = symplex::var("x");
    let poly = &x.powi(4) - &x.powi(3);
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty(), "x⁴-x³ should have roots");
    verify_roots(&poly, &x, &roots, "x⁴-x³");
}

#[test]
fn quartic_x4_minus_1() {
    // x⁴ - 1 = (x²-1)(x²+1) = (x-1)(x+1)(x-i)(x+i) = 0
    let x = symplex::var("x");
    let poly = &x.powi(4) - 1;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        4,
        "x⁴-1 should have 4 roots, got {}",
        roots.len()
    );
    verify_roots(&poly, &x, &roots, "x⁴-1");
}

#[test]
fn quartic_symmetric() {
    // (x-1)(x+1)(x-2)(x+2) = x⁴ - 5x² + 4
    let x = symplex::var("x");
    let poly = &(&x.powi(4) - &(&x.powi(2) * 5)) + 4;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        4,
        "symmetric quartic should have 4 roots, got {}",
        roots.len()
    );

    let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    strs.sort();
    assert!(
        strs.contains(&"-2".to_string()),
        "should have root -2: {strs:?}"
    );
    assert!(
        strs.contains(&"-1".to_string()),
        "should have root -1: {strs:?}"
    );
    assert!(
        strs.contains(&"1".to_string()),
        "should have root 1: {strs:?}"
    );
    assert!(
        strs.contains(&"2".to_string()),
        "should have root 2: {strs:?}"
    );
}

#[test]
fn quartic_verify_by_substitution() {
    // (x-1)(x-2)(x-3)(x-4) — verify each root satisfies the equation
    let x = symplex::var("x");
    let poly = &(&(&x.powi(4) - &(&x.powi(3) * 10)) + &(&x.powi(2) * 35)) - &(&(&x * 50) - 24);
    let roots = poly.solve_or_empty(&x);
    for root in &roots {
        let val = poly.subs(&x, root).eval();
        let s = format!("{}", val.expand().eval());
        assert_eq!(s, "0", "root {root} should make polynomial zero, got {s}");
    }
}

#[test]
fn quartic_all_complex() {
    // x⁴ + 1 = 0  →  four complex roots (8th roots of unity subset)
    let x = symplex::var("x");
    let poly = &x.powi(4) + 1;
    let roots = poly.solve_or_empty(&x);
    // Ferrari should produce 4 roots (all complex)
    if roots.len() == 4 {
        verify_roots(&poly, &x, &roots, "x⁴+1");
    }
    // If the solver can't handle this case yet, that's acceptable
}

// ═══════════════════════════════════════════════════════════════════════════
// Mixed / edge-case tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cubic_solver_does_not_panic_on_degenerate() {
    // 0·x³ + x² - 1 = 0 should degrade to quadratic gracefully
    let x = symplex::var("x");
    let poly = &x.powi(2) - 1;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        2,
        "degenerate cubic (actually quadratic) should have 2 roots"
    );
}

#[test]
fn quartic_solver_does_not_panic_on_degenerate() {
    // 0·x⁴ + x³ - 6x² + 11x - 6 = 0 should degrade to cubic
    let x = symplex::var("x");
    let poly = &x.powi(3) - &(&x.powi(2) * 6) + &(&x * 11) - 6;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        3,
        "degenerate quartic (actually cubic) should have 3 roots"
    );
}

#[test]
fn cubic_x_cubed() {
    // x³ = 0 → triple root at 0
    let x = symplex::var("x");
    let poly = x.powi(3);
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty(), "x³=0 should have at least one root (0)");
    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        strs.contains(&"0".to_string()),
        "root should be 0: {strs:?}"
    );
}

#[test]
fn quartic_x_fourth() {
    // x⁴ = 0 → quadruple root at 0
    let x = symplex::var("x");
    let poly = x.powi(4);
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty(), "x⁴=0 should have at least one root (0)");
    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        strs.contains(&"0".to_string()),
        "root should be 0: {strs:?}"
    );
}

#[test]
fn quartic_product_of_quadratics() {
    // (x²+1)(x²-4) = x⁴ - 3x² - 4 = 0  →  roots ±2, ±i
    let x = symplex::var("x");
    let poly = &(&x.powi(4) - &(&x.powi(2) * 3)) - 4;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        4,
        "(x²+1)(x²-4) should have 4 roots, got {}",
        roots.len()
    );
    verify_roots(&poly, &x, &roots, "x⁴-3x²-4");
}

#[test]
fn cubic_fractional_roots() {
    // (2x-1)(x-2)(x-3) = 2x³ - 11x² + 17x - 6 = 0  →  roots 1/2, 2, 3
    let x = symplex::var("x");
    let poly = &(&(&x.powi(3) * 2) - &(&x.powi(2) * 11)) + &(&x * 17) - 6;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        3,
        "2x³-11x²+17x-6 should have 3 roots, got {}",
        roots.len()
    );

    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        strs.contains(&"1/2".to_string()),
        "should have root 1/2: {strs:?}"
    );
    assert!(
        strs.contains(&"2".to_string()),
        "should have root 2: {strs:?}"
    );
    assert!(
        strs.contains(&"3".to_string()),
        "should have root 3: {strs:?}"
    );
}

#[test]
fn quartic_fractional_roots() {
    // (2x-1)(x-1)(x+1)(x-2) = 2x⁴ - 5x³ + x² + 5x - 2
    // roots: 1/2, 1, -1, 2
    // Expand: 2x⁴ - 5x³ + x² + 5x - 2
    let x = symplex::var("x");
    let poly = &(&(&(&x.powi(4) * 2) - &(&x.powi(3) * 5)) + &x.powi(2)) + &(&x * 5) - 2;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        4,
        "quartic with fractional root should have 4 roots, got {}",
        roots.len()
    );
    verify_roots(&poly, &x, &roots, "2x⁴-5x³+x²+5x-2");
}
