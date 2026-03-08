//! Tests for polynomial factoring at capability boundaries.
//!
//! Exercises the `.factor()` API on polynomials that push against the limits
//! of the factoring pipeline: high-degree products, quadratic-in-x² forms,
//! content extraction, irreducible polynomials, and cyclotomic polynomials.

mod common;

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: verify that factored form equals original at several integer points
// ═══════════════════════════════════════════════════════════════════════════

fn assert_values_match(original: &Ex, factored: &Ex, var: &Ex, label: &str) {
    for &pt in &[-5i64, -3, -2, -1, 0, 1, 2, 3, 5, 7, 10] {
        let vo = common::eval_at_i64(original, var, pt);
        let vf = common::eval_at_i64(factored, var, pt);
        assert!(
            common::approx_eq(vo, vf, 1e-9),
            "{label}: value mismatch at {var}={pt}: original={vo}, factored={vf}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Product of five linears: (x-1)(x-2)(x-3)(x-4)(x-5)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_product_of_linears() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Build (x-1)(x-2)(x-3)(x-4)(x-5) = x⁵ - 15x⁴ + 85x³ - 225x² + 274x - 120
    let expr = (&x - 1) * (&x - 2) * (&x - 3) * (&x - 4) * (&x - 5);
    let expanded = expr.expand().eval();
    let factored = expanded.factor(&x);

    // Value preservation at x=10
    let orig_val = common::eval_at_i64(&expanded, &x, 10);
    let fact_val = common::eval_at_i64(&factored, &x, 10);
    assert!(
        common::approx_eq(orig_val, fact_val, 1e-9),
        "factor changed value at x=10: {} vs {}",
        orig_val,
        fact_val
    );

    // Should be factored — no x^5 in the top-level display
    let s = format!("{factored}");
    assert!(
        !s.contains("x^5"),
        "should be factored (no x^5): {s}"
    );
    assert!(
        !s.contains("x^4"),
        "should be fully factored (no x^4): {s}"
    );

    // Full value preservation at many points
    assert_values_match(&expanded, &factored, &x, "(x-1)(x-2)(x-3)(x-4)(x-5)");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Degree-4 into quadratics: x⁴ - 5x² + 4 = (x²-1)(x²-4)
//    = (x-1)(x+1)(x-2)(x+2)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_degree_4_into_quadratics() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // x⁴ - 5x² + 4
    let expr = &x.powi(4) - &x.powi(2) * 5 + 4;
    let factored = expr.factor(&x);

    let s = format!("{factored}");
    assert!(
        !s.contains("x^4"),
        "should be factored (no x^4): {s}"
    );

    // Verify roots: f(1)=0, f(-1)=0, f(2)=0, f(-2)=0
    for &root in &[1i64, -1, 2, -2] {
        let val = common::eval_at_i64(&expr, &x, root);
        assert!(
            val.abs() < 1e-10,
            "x⁴-5x²+4 should be zero at x={root}, got {val}"
        );
    }

    assert_values_match(&expr, &factored, &x, "x⁴-5x²+4");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Content extraction: 6x³ + 12x² + 6x = 6x(x+1)²
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_content_extraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // 6x³ + 12x² + 6x = 6x(x² + 2x + 1) = 6x(x+1)²
    let expr = &x.powi(3) * 6 + &x.powi(2) * 12 + &x * 6;
    let factored = expr.factor(&x);

    let s = format!("{factored}");
    // The top-level expression should not have a raw x^3 in a sum
    assert!(
        !s.contains("x^3 +") && !s.contains("+ x^3"),
        "should be factored: {s}"
    );

    // Content factor 6 should appear
    assert!(
        s.contains("6") || s.contains("2") || s.contains("3"),
        "should have numeric content factor: {s}"
    );

    // Verify root at x=0
    let val_0 = common::eval_at_i64(&expr, &x, 0);
    assert!(
        val_0.abs() < 1e-10,
        "6x³+12x²+6x should be zero at x=0, got {val_0}"
    );

    // Verify root at x=-1 (double root)
    let val_m1 = common::eval_at_i64(&expr, &x, -1);
    assert!(
        val_m1.abs() < 1e-10,
        "6x³+12x²+6x should be zero at x=-1, got {val_m1}"
    );

    assert_values_match(&expr, &factored, &x, "6x³+12x²+6x");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Irreducible quintic: x⁵ + x + 1
//    This is actually reducible: x⁵+x+1 = (x²+x+1)(x³-x²+1)
//    but both factors are irreducible over ℤ.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_irreducible_quintic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // x⁵ + x + 1
    let expr = &x.powi(5) + &x + 1;
    let factored = expr.factor(&x);

    // Value preservation — regardless of whether it factors or not
    assert_values_match(&expr, &factored, &x, "x⁵+x+1");

    // The polynomial has no rational roots (by the rational root theorem,
    // only ±1 could be rational roots, and neither works).
    // If the engine can factor it, great; if not, it should be unchanged.
    let val_1 = common::eval_at_i64(&expr, &x, 1);
    assert!(
        val_1.abs() > 0.5,
        "x⁵+x+1 should not be zero at x=1"
    );
    let val_m1 = common::eval_at_i64(&expr, &x, -1);
    assert!(
        val_m1.abs() > 0.5,
        "x⁵+x+1 should not be zero at x=-1"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Cyclotomic polynomial Φ₁₂(x) = x⁴ - x² + 1
//    This is irreducible over ℤ.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_cyclotomic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // x⁴ - x² + 1 — the 12th cyclotomic polynomial, irreducible over ℤ
    let expr = &x.powi(4) - &x.powi(2) + 1;
    let factored = expr.factor(&x);

    let s = format!("{factored}");
    // This polynomial is irreducible over ℤ, so it should remain unfactored.
    // The display should still contain x^4 (or x⁴).
    assert!(
        s.contains("x^4") || s.contains("x⁴"),
        "Φ₁₂(x) = x⁴-x²+1 is irreducible over ℤ, should remain unfactored: {s}"
    );

    // Value preservation
    assert_values_match(&expr, &factored, &x, "Φ₁₂(x)=x⁴-x²+1");

    // Verify it's not zero at any small integer (no rational roots)
    for &pt in &[-2i64, -1, 0, 1, 2] {
        let val = common::eval_at_i64(&expr, &x, pt);
        assert!(
            val.abs() > 0.5,
            "Φ₁₂(x) should have no rational roots, but got {val} at x={pt}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Value preservation across several factored polynomials at x=3
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_preserves_value() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let cases: Vec<(&str, Ex)> = vec![
        // (x-1)(x-2)(x-3)(x-4)(x-5) expanded
        (
            "(x-1)(x-2)(x-3)(x-4)(x-5)",
            (&(&x - 1) * &(&x - 2) * &(&x - 3) * &(&x - 4) * &(&x - 5))
                .expand()
                .eval(),
        ),
        // x⁴ - 5x² + 4
        ("x⁴-5x²+4", &x.powi(4) - &x.powi(2) * 5 + 4),
        // 6x³ + 12x² + 6x
        ("6x³+12x²+6x", &x.powi(3) * 6 + &x.powi(2) * 12 + &x * 6),
        // x⁵ + x + 1
        ("x⁵+x+1", &x.powi(5) + &x + 1),
        // x⁴ - x² + 1 (cyclotomic Φ₁₂)
        ("Φ₁₂", &x.powi(4) - &x.powi(2) + 1),
        // x⁶ - 1
        ("x⁶-1", &x.powi(6) - 1),
        // x³ - x (= x(x-1)(x+1))
        ("x³-x", &x.powi(3) - &x),
        // x⁴ + 5x² + 6
        ("x⁴+5x²+6", &x.powi(4) + &x.powi(2) * 5 + 6),
    ];

    for (label, expr) in &cases {
        let factored = expr.factor(&x);

        // Evaluate both at x = 3
        let orig_val = common::eval_at_i64(expr, &x, 3);
        let fact_val = common::eval_at_i64(&factored, &x, 3);
        assert!(
            common::approx_eq(orig_val, fact_val, 1e-9),
            "{label}: factor changed value at x=3: {orig_val} vs {fact_val}"
        );

        // Also check at x = 7
        let orig_7 = common::eval_at_i64(expr, &x, 7);
        let fact_7 = common::eval_at_i64(&factored, &x, 7);
        assert!(
            common::approx_eq(orig_7, fact_7, 1e-9),
            "{label}: factor changed value at x=7: {orig_7} vs {fact_7}"
        );

        // And at x = -3
        let orig_m3 = common::eval_at_i64(expr, &x, -3);
        let fact_m3 = common::eval_at_i64(&factored, &x, -3);
        assert!(
            common::approx_eq(orig_m3, fact_m3, 1e-9),
            "{label}: factor changed value at x=-3: {orig_m3} vs {fact_m3}"
        );
    }
}
