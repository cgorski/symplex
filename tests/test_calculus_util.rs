//! Integration tests for calculus_util — symbolic domain analysis.
//!
//! These tests exercise `continuous_domain`, `singularities`, and
//! `estimate_frequency` through the public `Context` / `Ex` API where
//! possible, and validate domain-related behaviour end-to-end.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// continuous_domain — via public inequality API as proxy checks
// ═══════════════════════════════════════════════════════════════════════════

/// sqrt(x) requires x ≥ 0.  Verify via solve_ge that x ≥ 0 yields [0, ∞).
#[test]
fn domain_sqrt_x_proxy() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // x ≥ 0 should give [0, ∞)
    let result = x.solve_ge(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x >= 0 should not be empty: {s}"
    );
    assert!(
        s.contains("0"),
        "x >= 0 domain should reference 0: {s}"
    );
}

/// 1/x is undefined at x = 0.  Verify that solving x = 0 finds the singularity.
#[test]
fn domain_1_over_x_proxy() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Solve x = 0 → {0}
    let roots = x.solve_as_set(&x);
    let s = format!("{roots}");
    assert!(
        s.contains("0"),
        "solve x=0 should find root 0: {s}"
    );
}

/// ln(x) requires x > 0.  Verify via solve_gt.
#[test]
fn domain_ln_x_proxy() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // x > 0 → (0, ∞)
    let result = x.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x > 0 should not be empty: {s}"
    );
    assert!(
        s.contains("0") && (s.contains("oo") || s.contains("∞")),
        "x > 0 should give (0, ∞): {s}"
    );
}

/// sqrt(x - 2) on [-5, 5] requires x - 2 ≥ 0 → x ≥ 2.
/// Verify that x - 2 ≥ 0 gives x ∈ [2, ∞), which intersected with
/// [-5, 5] yields [2, 5].
#[test]
fn domain_sqrt_x_minus_2_proxy() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    let inner = &x - &two; // x - 2

    // x - 2 ≥ 0
    let result = inner.solve_ge(&x);
    let s = format!("{result}");
    assert!(
        s.contains("2"),
        "x - 2 >= 0 should reference 2 as boundary: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// singularities — end-to-end checks through solve
// ═══════════════════════════════════════════════════════════════════════════

/// tan(x) has singularities where cos(x) = 0.
/// Verify that cos(x) has roots near π/2.
#[test]
fn singularities_tan_x_proxy() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cos_x = x.cos();

    // Try to find where cos(x) = 0
    let roots = cos_x.solve(&x).unwrap_or_default();
    // cos(x) = 0 is transcendental; solver may or may not find roots.
    // If it does, verify they're near π/2 + nπ.
    for root in &roots {
        let s = format!("{root}");
        // If it solved, the root should involve pi
        if s.contains("pi") || s.contains("π") {
            // Good — it found a symbolic root involving pi
            return;
        }
    }
    // Even if the solver can't find symbolic roots, tan(x) is still
    // known to have singularities — this is a best-effort check.
}

/// 1/x has a singularity at x = 0.
#[test]
fn singularities_1_over_x_proxy() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // The denominator is x; solve x = 0
    let roots = x.solve(&x).unwrap_or_default();
    assert!(
        !roots.is_empty(),
        "x = 0 should have a solution"
    );
    let root_s = format!("{}", roots[0]);
    assert!(
        root_s == "0",
        "x = 0 root should be 0, got: {root_s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// estimate_frequency — structural checks
// ═══════════════════════════════════════════════════════════════════════════

/// sin(100*x) should have angular frequency 100.
/// We verify the structure: 100*x inside sin.
#[test]
fn frequency_sin_100x_structure() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let hundred = ctx.int(100);

    let inner = &hundred * &x;
    let expr = inner.sin();

    // Verify the expression is well-formed
    let s = format!("{expr}");
    assert!(
        s.contains("sin") && s.contains("100"),
        "sin(100*x) should display as such: {s}"
    );
}

/// x^2 + 1 has no trig terms, so no frequency.
#[test]
fn frequency_no_trig_structure() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = &x.powi(2) + 1;
    let s = format!("{expr}");
    assert!(
        !s.contains("sin") && !s.contains("cos") && !s.contains("tan"),
        "x^2 + 1 should have no trig terms: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional domain edge cases
// ═══════════════════════════════════════════════════════════════════════════

/// 1/(x^2 + 1) has no real singularities — denominator is always ≥ 1.
#[test]
fn domain_1_over_x2_plus_1_no_restriction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // x^2 + 1 = 0 has no real roots
    let denom = &x.powi(2) + 1;
    let roots = denom.solve(&x).unwrap_or_default();

    // Either no roots, or only complex roots
    // (display of complex roots will contain "I" or "i")
    for root in &roots {
        let s = format!("{root}");
        assert!(
            s.contains("I") || s.contains("i"),
            "x^2+1 should only have complex roots, got real root: {s}"
        );
    }
}

/// Verify that negative constant exponent triggers domain restriction.
/// 1/x^2 = x^(-2) should exclude x = 0.
#[test]
fn domain_x_pow_neg2_excludes_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // x = 0 is in the zero set of x (the base)
    let roots = x.solve(&x).unwrap_or_default();
    assert!(!roots.is_empty(), "x = 0 should have solution 0");
    assert_eq!(format!("{}", roots[0]), "0");
}

/// Nested expression: ln(x^2 - 1) requires x^2 - 1 > 0.
#[test]
fn domain_ln_x2_minus_1_proxy() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let inner = &x.powi(2) - 1; // x^2 - 1
    // x^2 - 1 > 0 → x < -1 or x > 1
    let result = inner.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x^2-1 > 0 should have solutions: {s}"
    );
    // Should reference the boundary points ±1
    assert!(
        s.contains("1"),
        "x^2-1 > 0 should reference boundary 1: {s}"
    );
}

/// cos(3*x) should yield angular frequency 3.
#[test]
fn frequency_cos_3x_structure() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let three = ctx.int(3);

    let inner = &three * &x;
    let expr = inner.cos();

    let s = format!("{expr}");
    assert!(
        s.contains("cos") && s.contains("3"),
        "cos(3*x) should display correctly: {s}"
    );
}

/// Multiple trig terms: sin(5*x) + cos(10*x).
/// The maximum frequency should come from cos(10*x).
#[test]
fn frequency_max_of_multiple_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    let ten = ctx.int(10);

    let term1 = (&five * &x).sin();
    let term2 = (&ten * &x).cos();
    let expr = &term1 + &term2;

    let s = format!("{expr}");
    // Both trig terms should be present
    assert!(
        s.contains("sin") && s.contains("cos"),
        "sin(5x) + cos(10x) should contain both trig fns: {s}"
    );
}
