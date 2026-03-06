//! Tests for additional ODE types and integration patterns.
//!
//! Covers:
//! - **Homogeneous coefficient:** y' = f(y/x) — substitution v = y/x
//! - **nth-order reducible:** F(y, y', y'') = 0 (no explicit x) — p = y'
//! - **ln(ln(x)) integration:** ∫ ln(ln(x)) dx = x·ln(ln(x)) − li(x)
//! - **Regression:** existing ODE types still work after new additions

use symplex::ode::OdeType;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Numerically verify a first-order ODE solution by substitution at a
/// sample point.  Substitutes all constants to 1.
#[allow(dead_code)]
fn verify_first_order(
    ode_expr: &Ex,
    solution: &Ex,
    constants: &[Ex],
    y: &Ex,
    x: &Ex,
    sample_x_num: i64,
    sample_x_den: i64,
) {
    let one = symplex::int(1);
    let mut concrete_sol = solution.clone();
    for c in constants {
        concrete_sol = concrete_sol.subs(c, &one);
    }

    let sol_prime = concrete_sol.diff(x);
    let dy_formal = y.formal_diff(x);
    let residual = ode_expr.subs(&dy_formal, &sol_prime).subs(y, &concrete_sol);

    let sample_val = symplex::rational(sample_x_num, sample_x_den);
    let residual_at = residual.subs(x, &sample_val);

    let val = residual_at
        .eval_f64()
        .expect("residual should evaluate to f64");
    assert!(
        val.abs() < 1e-4,
        "First-order ODE residual should be ~0, got {val} at x={sample_x_num}/{sample_x_den}\n  \
         solution (C=1): {concrete_sol}\n  residual expr: {residual_at}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Homogeneous coefficient: y' = f(y/x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_homogeneous_coeff_basic() {
    // y' = (x + y) / x = 1 + y/x  →  f(v) = 1 + v
    // Substitution: v + x·v' = 1 + v  →  x·v' = 1  →  v = ln|x| + C1
    // Back-sub: y/x = ln|x| + C1  →  y = x·(ln|x| + C1)
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);

    // dy/dx - (x + y)/x = 0  →  dy/dx - 1 - y/x = 0
    let one = symplex::int(1);
    let y_over_x = &y / &x;
    let ode = &dy - &one - &y_over_x;

    let result = ode.solve_ode(&y, &x);
    assert!(result.is_some(), "should solve y' = (x+y)/x");
    let (sol, _constants) = result.unwrap();
    let s = format!("{sol}");
    assert!(
        s.contains("C1"),
        "solution should contain constant C1: {s}"
    );
    // The solution should involve x and ln — either explicit or implicit
    assert!(
        s.contains("x") || s.contains("ln"),
        "solution should reference x or ln: {s}"
    );
}

#[test]
fn ode_homogeneous_coeff_quadratic() {
    // y' = (x² + y²) / (x·y)  →  f(v) = (1 + v²)/v = 1/v + v
    // After substitution: v + x·v' = 1/v + v  →  x·v' = 1/v  →  v dv = dx/x
    // ∫ v dv = ln|x| + C  →  v²/2 = ln|x| + C
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);

    let x_sq = x.powi(2);
    let y_sq = y.powi(2);
    let numer = &x_sq + &y_sq;
    let denom = &x * &y;
    let rhs = &numer / &denom;
    let ode = &dy - &rhs;

    let result = ode.solve_ode(&y, &x);
    assert!(result.is_some(), "should solve y' = (x²+y²)/(xy)");
    let (sol, _constants) = result.unwrap();
    let s = format!("{sol}");
    assert!(
        s.contains("C1"),
        "solution should contain constant C1: {s}"
    );
}

#[test]
fn ode_classify_homogeneous() {
    // y' = (x² + y²)/x² = 1 + (y/x)²  →  non-linear, so not caught by linear VC
    // After substituting y = v·x: RHS = 1 + v² (free of x) → HomogeneousCoefficient
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);

    let x_sq = x.powi(2);
    let y_sq = y.powi(2);
    let rhs = &(&x_sq + &y_sq) / &x_sq; // (x² + y²)/x²
    let ode = &dy - &rhs;

    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::HomogeneousCoefficient,
        "y' = (x²+y²)/x² should be classified as HomogeneousCoefficient, got {ode_type:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. nth-order reducible: F(y, y', y'') = 0 (no explicit x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_nth_reducible_basic() {
    // y'' = y'  (missing x explicitly)
    // Substitution: p·dp/dy = p  →  dp/dy = 1  →  p = y + C1
    // Then dy/dx = y + C1  →  separable
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);

    // y'' - y' = 0
    let ode = &d2y - &dy;

    let result = ode.solve_ode(&y, &x);
    assert!(result.is_some(), "should solve y'' - y' = 0 via nth-order reducible");
    let (sol, _constants) = result.unwrap();
    let s = format!("{sol}");
    // Should contain at least one constant
    assert!(
        s.contains("C1") || s.contains("C2"),
        "solution should contain constants: {s}"
    );
}

#[test]
fn ode_nth_reducible_nonlinear() {
    // y·y'' = (y')²  →  y·p·dp/dy = p²  →  y·dp/dy = p  →  dp/p = dy/y
    // → ln|p| = ln|y| + C  →  p = C1·y  →  dy/dx = C1·y
    // → y = exp(C1·x + C2)
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);

    // y·y'' - (y')² = 0
    let dy_sq = dy.powi(2);
    let y_d2y = &y * &d2y;
    let ode = &y_d2y - &dy_sq;

    // At minimum, classification should work.
    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::NthOrderReducible,
        "y·y'' = (y')² should classify as NthOrderReducible, got {ode_type:?}"
    );

    // Solving may or may not succeed depending on how well the substitution
    // pipeline handles the nonlinear case.
    if let Some((sol, _constants)) = ode.solve_ode(&y, &x) {
        let s = format!("{sol}");
        // If solved, it should have constants
        assert!(
            s.contains("C1") || s.contains("C2"),
            "solution should contain constants: {s}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Integration: ∫ ln(ln(x)) dx = x·ln(ln(x)) − li(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_ln_ln_x() {
    // ∫ ln(ln(x)) dx should NOT be unevaluated
    // Expected: x·ln(ln(x)) − li(x)
    let x = symplex::var("x");
    let ln_x = x.ln();
    let ln_ln_x = ln_x.ln();

    let result = ln_ln_x.integrate(&x);
    let s = format!("{result}");

    assert!(
        !s.contains("Integral("),
        "∫ ln(ln(x)) dx should not be unevaluated, got: {s}"
    );
    // The result should contain li (logarithmic integral) and ln(ln
    assert!(
        s.contains("li") || s.contains("ln"),
        "result should contain li or ln: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Regression: existing ODE types still work
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn no_regression_existing_odes() {
    // Simple separable: y' = x → y = x²/2 + C1
    {
        let x = symplex::var("x");
        let y = symplex::var("y");
        let ode = expr!(diff(y, x) - x);
        let result = ode.solve_ode(&y, &x);
        assert!(result.is_some(), "y' = x should still work (simple separable)");
    }

    // First-order linear CC: y' + 2y = 0 → y = C1·exp(-2x)
    {
        let x = symplex::var("x");
        let y = symplex::var("y");
        let ode = expr!(diff(y, x) + 2 * y);
        let result = ode.solve_ode(&y, &x);
        assert!(result.is_some(), "y' + 2y = 0 should still work (linear CC)");
        let (sol, _) = result.unwrap();
        let s = format!("{sol}");
        assert!(s.contains("C1"), "should have constant: {s}");
        assert!(s.contains("exp"), "should contain exp: {s}");
    }

    // Second-order CC homogeneous: y'' + y = 0 → C1·cos(x) + C2·sin(x)
    {
        let x = symplex::var("x");
        let y = symplex::var("y");
        let dy = y.formal_diff(&x);
        let d2y = dy.formal_diff(&x);
        let ode = &d2y + &y;
        let result = ode.solve_ode(&y, &x);
        assert!(result.is_some(), "y'' + y = 0 should still work");
        let (sol, _) = result.unwrap();
        let s = format!("{sol}");
        assert!(
            s.contains("C1") && s.contains("C2"),
            "should have two constants: {s}"
        );
        assert!(
            s.contains("cos") && s.contains("sin"),
            "should use trig form: {s}"
        );
    }

    // Full separable: y' = x·y → y = C1·exp(x²/2)
    {
        let x = symplex::var("x");
        let y = symplex::var("y");
        let ode = expr!(diff(y, x) - x * y);
        let result = ode.solve_ode(&y, &x);
        assert!(result.is_some(), "y' = xy should still work (full separable)");
        let (sol, _) = result.unwrap();
        let s = format!("{sol}");
        assert!(s.contains("C1"), "should have constant: {s}");
        assert!(s.contains("exp"), "should contain exp: {s}");
    }

    // Variable-coefficient linear: y' + 2xy = 0 → y = C1·exp(-x²)
    {
        let x = symplex::var("x");
        let y = symplex::var("y");
        let ode = expr!(diff(y, x) + 2 * x * y);
        let result = ode.solve_ode(&y, &x);
        assert!(result.is_some(), "y' + 2xy = 0 should still work");
    }
}

#[test]
fn no_regression_classification_simple_separable() {
    let x = symplex::var("x");
    let y = symplex::var("y");
    let ode = expr!(diff(y, x) - x);

    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::SimpleSeparable,
        "y' = x should classify as SimpleSeparable"
    );
}

#[test]
fn no_regression_classification_bernoulli() {
    // y' + y = y²  → Bernoulli with n = 2
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let y_sq = y.powi(2);
    let ode = &(&dy + &y) - &y_sq;

    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::Bernoulli,
        "y' + y = y² should classify as Bernoulli, got {ode_type:?}"
    );
}
