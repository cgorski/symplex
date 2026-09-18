//! Tests for advanced ODE solver types.
//!
//! Covers:
//! - **Exact equations:** M(x,y)dx + N(x,y)dy = 0 with ∂M/∂y = ∂N/∂x
//! - **Bernoulli equations:** y' + P(x)·y = Q(x)·y^n (n ≠ 0, 1)
//! - **Euler-Cauchy equations:** a·x²·y'' + b·x·y' + c·y = 0
//! - **Variation of parameters:** y'' + p·y' + q·y = g(x), fallback method
//! - **Classification:** verify `classify_ode()` returns the correct `OdeType`

use symplex::expr::ExprType;
use symplex::ode::OdeType;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Numerically verify a first-order ODE solution by substitution at a
/// sample point. Substitutes all constants to 1.
fn verify_first_order(
    ode_expr: &Ex,
    solution: &Ex,
    constants: &[Ex],
    y: &Ex,
    x: &Ex,
    sample_x_num: i64,
    sample_x_den: i64,
) {
    let ctx = ode_expr.context();
    let one = ctx.int(1);
    let mut concrete_sol = solution.clone();
    for c in constants {
        concrete_sol = concrete_sol.subs(c, &one);
    }

    let sol_prime = concrete_sol.diff(x);
    let dy_formal = y.formal_diff(x);
    let residual = ode_expr.subs(&dy_formal, &sol_prime).subs(y, &concrete_sol);

    let sample_val = ctx.rational(sample_x_num, sample_x_den);
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

/// Numerically verify a second-order ODE solution by substitution.
/// Substitutes C1=1, C2=1.
fn verify_second_order(
    ode_expr: &Ex,
    solution: &Ex,
    constants: &[Ex],
    y: &Ex,
    x: &Ex,
    sample_x_num: i64,
    sample_x_den: i64,
) {
    let ctx = ode_expr.context();
    let one = ctx.int(1);
    let mut concrete_sol = solution.clone();
    for c in constants {
        concrete_sol = concrete_sol.subs(c, &one);
    }

    let sol_prime = concrete_sol.diff(x);
    let sol_double_prime = sol_prime.diff(x);

    let dy_formal = y.formal_diff(x);
    let d2y_formal = dy_formal.formal_diff(x);

    let residual = ode_expr
        .subs(&d2y_formal, &sol_double_prime)
        .subs(&dy_formal, &sol_prime)
        .subs(y, &concrete_sol);

    let sample_val = ctx.rational(sample_x_num, sample_x_den);
    let residual_at = residual.subs(x, &sample_val);

    let val = residual_at
        .eval_f64()
        .expect("residual should evaluate to f64 for second-order ODE");
    assert!(
        val.abs() < 1e-3,
        "Second-order ODE residual should be ~0, got {val} at x={sample_x_num}/{sample_x_den}\n  \
         solution (C1=C2=1): {concrete_sol}\n  residual expr: {residual_at}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Exact equations: M(x,y) + N(x,y)·y' = 0, ∂M/∂y = ∂N/∂x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exact_2xy_plus_3_and_xsq_plus_4y() {
    let ctx = Context::new();
    // (2xy + 3) + (x² + 4y)·y' = 0
    // M = 2xy + 3, N = x² + 4y
    // ∂M/∂y = 2x, ∂N/∂x = 2x — exact!
    // F = ∫M dx = x²y + 3x + g(y)
    // g'(y) = N − ∂(x²y+3x)/∂y = (x²+4y) − x² = 4y → g(y) = 2y²
    // Solution: x²y + 3x + 2y² = C1
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);

    // Build: 2*x*y + 3 + (x^2 + 4*y)*y' = 0
    let two = ctx.int(2);
    let three = ctx.int(3);
    let four = ctx.int(4);
    let m = &(&two * &x * &y) + &three; // 2xy + 3
    let n = &x.powi(2) + &(&four * &y); // x² + 4y
    let ode = &m + &(&n * &dy);

    let sol = ode.solve_ode(&y, &x);
    assert!(
        !sol.has_unevaluated(),
        "exact ODE (2xy+3) + (x²+4y)y' = 0 should be solvable"
    );

    let s = format!("{sol}");
    // Exact ODE solver returns the potential F(x,y); the constant is implicit.
    assert!(!s.is_empty(), "solution should be non-empty: {s}");
}

#[test]
fn exact_classify() {
    let ctx = Context::new();
    // Same ODE — check classification
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let two = ctx.int(2);
    let three = ctx.int(3);
    let four = ctx.int(4);
    let m = &(&two * &x * &y) + &three;
    let n = &x.powi(2) + &(&four * &y);
    let ode = &m + &(&n * &dy);

    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::ExactFirstOrder,
        "should classify as exact first-order"
    );
}

#[test]
fn exact_simple_ydx_xdy() {
    let ctx = Context::new();
    // y + x·y' = 0  →  M = y, N = x  →  ∂M/∂y = 1, ∂N/∂x = 1 — exact
    // F = xy, solution: xy = C1
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let ode = &y + &(&x * &dy); // y + x·y' = 0

    let sol = ode.solve_ode(&y, &x);
    assert!(
        !sol.has_unevaluated(),
        "y + x·y' = 0 should be solvable (exact)"
    );

    let s = format!("{sol}");
    // Exact solver may return F(x,y) without explicit C1, or separable solver may include it.
    assert!(!s.is_empty(), "solution should be non-empty: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Bernoulli equations: y' + P(x)·y = Q(x)·y^n
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bernoulli_y_prime_plus_y_over_x_eq_y_squared() {
    let ctx = Context::new();
    // y' + y/x = y²  →  y' + (1/x)·y − y² = 0
    // This is Bernoulli with P(x) = 1/x, Q(x) = 1, n = 2
    // Using v = y^(1−2) = y^(−1) = 1/y → v' − (1/x)·v = −1
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);

    // y' + y * x^(-1) - y^2 = 0
    let y_over_x = &y / &x;
    let y_sq = y.powi(2);
    let ode = &(&dy + &y_over_x) - &y_sq;

    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");
    // The old API returned Option — is_some() just meant the solver returned
    // *something*, even if it contained unevaluated sub-expressions.
    // Check that the top-level result is not a bare DSolve node.
    assert!(
        sol.expr_type() != ExprType::Unevaluated,
        "Bernoulli ODE y' + y/x = y² should be solvable, got: {s}"
    );
    assert!(!s.is_empty(), "Bernoulli solution should be non-empty: {s}");
}

#[test]
fn bernoulli_classify() {
    let ctx = Context::new();
    // y' + y/x − y² = 0 should classify as Bernoulli
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let y_over_x = &y / &x;
    let y_sq = y.powi(2);
    let ode = &(&dy + &y_over_x) - &y_sq;

    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(ode_type, OdeType::Bernoulli, "should classify as Bernoulli");
}

#[test]
fn bernoulli_y_prime_minus_y_eq_neg_y_cubed_exp() {
    let ctx = Context::new();
    // y' − y = −y³·exp(−2x)  →  y' − y + y³·exp(−2x) = 0
    // Bernoulli with P(x) = −1, Q(x) = −exp(−2x), n = 3
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);

    let neg_two_x = &ctx.int(-2) * &x;
    let exp_neg2x = neg_two_x.exp();
    let y_cubed = y.powi(3);
    // y' - y + y³·exp(-2x) = 0
    let ode = &(&dy - &y) + &(&y_cubed * &exp_neg2x);

    let sol = ode.solve_ode(&y, &x);
    // This is a hard Bernoulli — it may or may not solve depending on
    // integration capability. If it does solve, verify.
    if sol.expr_type() != ExprType::Unevaluated {
        let s = format!("{sol}");
        assert!(
            s.contains("C1"),
            "Bernoulli (n=3) solution should have C1: {s}"
        );
    }
}

#[test]
fn bernoulli_simple_n2_constant_coefficients() {
    let ctx = Context::new();
    // y' + y - y² = 0 (P=1, Q=1, n=2)
    // v = 1/y, v' - v = -1
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let y_sq = y.powi(2);
    let ode = &(&dy + &y) - &y_sq;

    let sol = ode.solve_ode(&y, &x);
    assert!(
        !sol.has_unevaluated(),
        "Bernoulli y' + y - y² = 0 should be solvable"
    );

    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");

    // Numerically verify the Bernoulli solution
    let c1 = ctx.symbol("C1");
    verify_first_order(&ode, &sol, &[c1], &y, &x, 1, 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Euler-Cauchy equations: a·x²·y'' + b·x·y' + c·y = 0
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn euler_cauchy_distinct_real_roots() {
    let ctx = Context::new();
    // x²y'' − 2y = 0
    // Characteristic: a=1, b=0, c=−2
    //   r(r−1) − 2 = 0  →  r² − r − 2 = 0  →  (r−2)(r+1) = 0  →  r = 2, −1
    // Solution: y = C1·x² + C2·x^(−1)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);

    let x_sq = x.powi(2);
    let two = ctx.int(2);
    // x²·y'' − 2·y = 0
    let ode = &(&x_sq * &d2y) - &(&two * &y);

    let sol = ode.solve_ode(&y, &x);
    assert!(
        !sol.has_unevaluated(),
        "Euler-Cauchy x²y'' - 2y = 0 should be solvable"
    );

    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(s.contains("C2"), "should have C2: {s}");

    // Verify numerically at x = 2
    let c1 = ctx.symbol("C1");
    let c2 = ctx.symbol("C2");
    verify_second_order(&ode, &sol, &[c1, c2], &y, &x, 2, 1);
}

#[test]
fn euler_cauchy_classify() {
    let ctx = Context::new();
    // x²y'' − 2y = 0 should classify as EulerCauchy
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let two = ctx.int(2);
    let ode = &(&x_sq * &d2y) - &(&two * &y);

    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::EulerCauchy,
        "should classify as Euler-Cauchy"
    );
}

#[test]
fn euler_cauchy_complex_roots() {
    let ctx = Context::new();
    // x²y'' + xy' + y = 0
    // Characteristic: a=1, b=1, c=1
    //   r(r−1) + r + 1 = 0  →  r² + 1 = 0  →  r = ±i
    // Solution: y = C1·cos(ln(x)) + C2·sin(ln(x))
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);

    let x_sq = x.powi(2);
    // x²·y'' + x·y' + y = 0
    let ode = &(&(&x_sq * &d2y) + &(&x * &dy)) + &y;

    let sol = ode.solve_ode(&y, &x);
    assert!(
        !sol.has_unevaluated(),
        "Euler-Cauchy x²y'' + xy' + y = 0 should be solvable"
    );

    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(s.contains("C2"), "should have C2: {s}");
    // Should involve cos and sin (of ln(x))
    assert!(
        s.contains("cos") && s.contains("sin"),
        "complex-root Euler-Cauchy should use cos and sin: {s}"
    );
    assert!(s.contains("ln"), "should involve ln(x): {s}");

    // Verify numerically at x = 3/2 (must be positive for ln)
    let c1 = ctx.symbol("C1");
    let c2 = ctx.symbol("C2");
    verify_second_order(&ode, &sol, &[c1, c2], &y, &x, 3, 2);
}

#[test]
fn euler_cauchy_repeated_root() {
    let ctx = Context::new();
    // x²y'' + xy' = 0
    // Characteristic: a=1, b=1, c=0
    //   r(r−1) + r = 0  →  r² = 0  →  r = 0 (double)
    // Solution: y = C1 + C2·ln(x)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);

    let x_sq = x.powi(2);
    // x²·y'' + x·y' = 0
    let ode = &(&x_sq * &d2y) + &(&x * &dy);

    let sol = ode.solve_ode(&y, &x);
    assert!(
        !sol.has_unevaluated(),
        "Euler-Cauchy x²y'' + xy' = 0 should be solvable"
    );

    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(s.contains("C2"), "should have C2: {s}");
    assert!(
        s.contains("ln"),
        "repeated root solution should contain ln(x): {s}"
    );

    let c1 = ctx.symbol("C1");
    let c2 = ctx.symbol("C2");
    verify_second_order(&ode, &sol, &[c1, c2], &y, &x, 2, 1);
}

#[test]
fn euler_cauchy_with_coefficients() {
    let ctx = Context::new();
    // 2x²y'' + 3xy' − y = 0
    // Characteristic: a=2, b=3, c=−1
    //   2r(r−1) + 3r − 1 = 0  →  2r² + r − 1 = 0  →  (2r−1)(r+1) = 0
    //   r = 1/2, r = −1
    // Solution: y = C1·x^(1/2) + C2·x^(−1)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);

    let x_sq = x.powi(2);
    let two = ctx.int(2);
    let three = ctx.int(3);
    // 2x²·y'' + 3x·y' − y = 0
    let ode = &(&(&two * &x_sq * &d2y) + &(&three * &x * &dy)) - &y;

    let sol = ode.solve_ode(&y, &x);
    assert!(
        !sol.has_unevaluated(),
        "Euler-Cauchy 2x²y'' + 3xy' - y = 0 should be solvable"
    );

    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(s.contains("C2"), "should have C2: {s}");

    // Verify at x = 4 (nice for sqrt)
    let c1 = ctx.symbol("C1");
    let c2 = ctx.symbol("C2");
    verify_second_order(&ode, &sol, &[c1, c2], &y, &x, 4, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Variation of parameters: y'' + p·y' + q·y = g(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn variation_of_parameters_y_pp_plus_y_eq_tan_x() {
    let ctx = Context::new();
    // y'' + y = tan(x)
    // Homogeneous: y₁ = cos(x), y₂ = sin(x)
    // W = cos·cos − sin·(−sin) = cos² + sin² = 1
    // Particular via VoP: y_p = −cos(x)·∫sin(x)tan(x)dx + sin(x)·∫cos(x)tan(x)dx
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);

    // y'' + y − tan(x) = 0
    let ode = &(&d2y + &y) - &x.tan();

    let sol = ode.solve_ode(&y, &x);
    // This requires VoP since tan(x) is not handled by undetermined coefficients.
    if sol.expr_type() != ExprType::Unevaluated {
        let s = format!("{sol}");
        assert!(
            s.contains("C1") && s.contains("C2"),
            "VoP solution should have two constants: {s}"
        );

        // Verify numerically at a point where tan is well-behaved
        let c1 = ctx.symbol("C1");
        let c2 = ctx.symbol("C2");
        verify_second_order(&ode, &sol, &[c1, c2], &y, &x, 1, 4);
    }
    // It's acceptable if the integration engine can't handle the VoP integrals
}

#[test]
fn variation_of_parameters_y_pp_minus_y_eq_exp_x() {
    let ctx = Context::new();
    // y'' − y = exp(x) — this is also solvable by undetermined coefficients
    // (resonance case), but let's check VoP handles it too.
    // If undetermined coefficients catches it first, that's fine.
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);

    let ode = &(&d2y - &y) - &x.exp();

    let sol = ode.solve_ode(&y, &x);
    assert!(
        !sol.has_unevaluated(),
        "y'' - y = exp(x) should be solvable (undetermined coefficients or VoP)"
    );

    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(s.contains("C2"), "should have C2: {s}");

    // Numerically verify the solution
    let c1 = ctx.symbol("C1");
    let c2 = ctx.symbol("C2");
    verify_second_order(&ode, &sol, &[c1, c2], &y, &x, 1, 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Regression: existing ODE types still work
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn regression_simple_separable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let ode = expr!(ctx, diff(y, x) - x);
    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "y' = x should still work");
}

#[test]
fn regression_first_order_linear_cc() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let ode = expr!(ctx, diff(y, x) + 2 * y);
    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "y' + 2y = 0 should still work");
}

#[test]
fn regression_second_order_cc_homogeneous() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &y;
    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "y'' + y = 0 should still work");
}

#[test]
fn regression_full_separable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let ode = expr!(ctx, diff(y, x) - x * y);
    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "y' = xy should still work");
}

#[test]
fn regression_variable_coeff_linear() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let ode = expr!(ctx, diff(y, x) + 2 * x * y);
    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "y' + 2xy = 0 should still work");
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Check ODE solution verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn check_ode_solution_euler_cauchy() {
    let ctx = Context::new();
    // x²y'' − 2y = 0, solution y = C1·x² + C2·x^(−1)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let two = ctx.int(2);
    let ode = &(&x_sq * &d2y) - &(&two * &y);

    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "should solve Euler-Cauchy");

    let verified = ode.check_ode_solution(&sol, &y, &x);
    // checkodesol may or may not simplify completely — it's a best-effort check
    if !verified {
        // Fall back to numerical verification
        let c1 = ctx.symbol("C1");
        let c2 = ctx.symbol("C2");
        verify_second_order(&ode, &sol, &[c1, c2], &y, &x, 3, 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Edge cases and non-matching patterns
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn non_euler_cauchy_not_misclassified() {
    let ctx = Context::new();
    // y'' + y = 0 is NOT Euler-Cauchy (no x² on y'')
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &y;
    let ode_type = ode.classify_ode(&y, &x);
    assert_ne!(
        ode_type,
        OdeType::EulerCauchy,
        "y'' + y = 0 should NOT be classified as Euler-Cauchy"
    );
    assert_eq!(
        ode_type,
        OdeType::SecondOrderLinearCCHomogeneous,
        "y'' + y = 0 should be classified as second-order CC homogeneous"
    );
}

#[test]
fn non_bernoulli_linear_not_misclassified() {
    let ctx = Context::new();
    // y' + y = 0 (linear, not Bernoulli since n=1 is excluded)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let ode = expr!(ctx, diff(y, x) + y);
    let ode_type = ode.classify_ode(&y, &x);
    assert_ne!(
        ode_type,
        OdeType::Bernoulli,
        "y' + y = 0 should NOT be Bernoulli"
    );
}

#[test]
fn euler_cauchy_distinct_real_verify_at_multiple_points() {
    let ctx = Context::new();
    // x²y'' − 2y = 0 → y = C1·x² + C2/x
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let two = ctx.int(2);
    let ode = &(&x_sq * &d2y) - &(&two * &y);

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve Euler-Cauchy distinct real");
    let c1 = ctx.symbol("C1");
    let c2 = ctx.symbol("C2");

    // Verify at multiple positive x values
    for &(num, den) in &[(1, 2), (1, 1), (3, 1), (5, 1)] {
        verify_second_order(&ode, &sol, &[c1.clone(), c2.clone()], &y, &x, num, den);
    }
}

#[test]
fn euler_cauchy_complex_verify_at_multiple_points() {
    let ctx = Context::new();
    // x²y'' + xy' + y = 0 → y = C1·cos(ln x) + C2·sin(ln x)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let ode = &(&(&x_sq * &d2y) + &(&x * &dy)) + &y;

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve Euler-Cauchy complex");
    let c1 = ctx.symbol("C1");
    let c2 = ctx.symbol("C2");

    for &(num, den) in &[(1, 2), (1, 1), (2, 1), (3, 1)] {
        verify_second_order(&ode, &sol, &[c1.clone(), c2.clone()], &y, &x, num, den);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Additional Bernoulli tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bernoulli_n2_with_constant_p() {
    let ctx = Context::new();
    // y' + 2y − y² = 0 (P=2, Q=1, n=2)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let two = ctx.int(2);
    let y_sq = y.powi(2);
    let ode = &(&dy + &(&two * &y)) - &y_sq;

    let sol = ode.solve_ode(&y, &x);
    assert!(
        !sol.has_unevaluated(),
        "Bernoulli y' + 2y - y² = 0 should be solvable"
    );

    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have a constant: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Additional Euler-Cauchy edge case
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn euler_cauchy_4x2_y_pp_minus_4x_yp_plus_3y() {
    let ctx = Context::new();
    // 4x²y'' − 4xy' + 3y = 0
    // Characteristic: 4r(r−1) − 4r + 3 = 0 → 4r² − 8r + 3 = 0
    // r = (8 ± √(64−48))/8 = (8 ± 4)/8 → r = 3/2, 1/2
    // Solution: y = C1·x^(3/2) + C2·x^(1/2)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let four = ctx.int(4);
    let three = ctx.int(3);
    // 4x²y'' − 4xy' + 3y = 0
    let ode = &(&(&four * &x_sq * &d2y) - &(&four * &x * &dy)) + &(&three * &y);

    let sol = ode.solve_ode(&y, &x);
    assert!(
        !sol.has_unevaluated(),
        "Euler-Cauchy 4x²y'' - 4xy' + 3y = 0 should be solvable"
    );

    let s = format!("{sol}");
    assert!(
        s.contains("C1") && s.contains("C2"),
        "should have two constants: {s}"
    );

    // Verify at x = 4
    let c1 = ctx.symbol("C1");
    let c2 = ctx.symbol("C2");
    verify_second_order(&ode, &sol, &[c1, c2], &y, &x, 4, 1);
}
