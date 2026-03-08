//! Tests for second-order constant-coefficient nonhomogeneous ODE solver
//! using the method of undetermined coefficients.
//!
//! Covers:
//! - Constant forcing: y'' + y = 1
//! - Linear forcing: y'' + y' + y = x
//! - Quadratic forcing: y'' + y = x²
//! - Numerical verification by substitution
//! - Regression: homogeneous and first-order still work
//! - Classification of nonhomogeneous ODEs
//! - Graceful failure for non-polynomial forcing (e.g., exp)

mod common;
use symplex::ode::OdeType;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Numerically verify a second-order ODE solution by substituting
/// C1=1, C2=1 into the solution, computing derivatives, replacing
/// formal derivatives and y in the ODE, and evaluating at a sample point.
fn verify_second_order_numerically(
    ode_expr: &Ex,
    solution: &Ex,
    constants: &[Ex],
    y: &Ex,
    x: &Ex,
    sample_x_num: i64,
    sample_x_den: i64,
) {
    let one = symplex::default_context().int(1);
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

    let sample_val = symplex::default_context().rational(sample_x_num, sample_x_den);
    let residual_at = residual.subs(x, &sample_val);

    let val = residual_at
        .eval_f64()
        .expect("evalf_f64 should succeed for second-order ODE residual");
    assert!(
        val.abs() < 1e-4,
        "Second-order ODE residual should be ~0, got {val} at x={sample_x_num}/{sample_x_den}\n  \
         solution (C=1): {concrete_sol}\n  residual: {residual_at}"
    );
}

/// Numerically verify a first-order ODE solution.
fn verify_first_order_numerically(
    ode_expr: &Ex,
    solution: &Ex,
    constants: &[Ex],
    y: &Ex,
    x: &Ex,
    sample_x_num: i64,
    sample_x_den: i64,
) {
    let one = symplex::default_context().int(1);
    let mut concrete_sol = solution.clone();
    for c in constants {
        concrete_sol = concrete_sol.subs(c, &one);
    }

    let sol_prime = concrete_sol.diff(x);

    let dy_formal = y.formal_diff(x);
    let residual = ode_expr
        .subs(&dy_formal, &sol_prime)
        .subs(y, &concrete_sol);

    let sample_val = symplex::default_context().rational(sample_x_num, sample_x_den);
    let residual_at = residual.subs(x, &sample_val);

    let val = residual_at
        .eval_f64()
        .expect("evalf_f64 should succeed for first-order ODE residual");
    assert!(
        val.abs() < 1e-6,
        "First-order ODE residual should be ~0, got {val} at x={sample_x_num}/{sample_x_den}\n  \
         solution (C=1): {concrete_sol}\n  residual: {residual_at}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: Constant forcing  y'' + y = 1
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_second_order_constant_rhs() {
    // y'' + y - 1 = 0  ⟹  y'' + y = 1
    // Particular solution: y_p = 1 (since c=1 ≠ 0, y_p = k/c = 1/1)
    // Homogeneous: y_h = C1·exp(ix) + C2·exp(-ix)
    // General: y = C1·exp(ix) + C2·exp(-ix) + 1
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let one = symplex::default_context().int(1);
    let ode = &d2y + &y - &one; // y'' + y - 1 = 0

    let result = ode.solve_ode(&y, &x);
    assert!(result.is_some(), "y'' + y = 1 should be solvable");

    let (sol, constants) = result.unwrap();
    let s = format!("{sol}");
    assert!(
        s.contains("C1") && s.contains("C2"),
        "should have two constants: {s}"
    );
    assert_eq!(constants.len(), 2, "should have exactly two constants");

    // The solution should contain exp terms (homogeneous) and the constant 1 (particular).
    // Verify numerically at a test point.
    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 7, 10);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Linear forcing  y'' + y' + y = x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_second_order_linear_rhs() {
    // y'' + y' + y - x = 0  ⟹  y'' + y' + y = x
    // With b=1, c=1, rhs = x  (degree 1)
    // Try y_p = Ax + B:  y_p''=0, y_p'=A
    //   A + Ax + B = x  ⟹  cA = 1 → A=1, bA + cB = 0 → 1 + B = 0 → B=-1
    //   y_p = x - 1
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &dy + &y - &x; // y'' + y' + y - x = 0

    let result = ode.solve_ode(&y, &x);
    assert!(result.is_some(), "y'' + y' + y = x should be solvable");

    let (sol, constants) = result.unwrap();
    let s = format!("{sol}");
    assert!(
        s.contains("C1") && s.contains("C2"),
        "should have two constants: {s}"
    );

    // Verify numerically
    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 1, 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Quadratic forcing  y'' + y = x²
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_second_order_quadratic_rhs() {
    // y'' + y - x² = 0  ⟹  y'' + y = x²
    // With b=0, c=1, rhs = x² (degree 2)
    // Try y_p = Ax² + Bx + D:
    //   y_p'' = 2A, y_p' = 2Ax + B
    //   2A + Ax² + Bx + D = x²
    //   A = 1, B = 0, 2A + D = 0 → D = -2
    //   y_p = x² - 2
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let ode = &d2y + &y - &x_sq; // y'' + y - x² = 0

    let result = ode.solve_ode(&y, &x);
    assert!(result.is_some(), "y'' + y = x² should be solvable");

    let (sol, constants) = result.unwrap();
    let s = format!("{sol}");
    assert!(
        s.contains("C1") && s.contains("C2"),
        "should have two constants: {s}"
    );

    // Verify numerically
    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 3, 10);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Numerical verification by substitution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_second_order_nonhomogeneous_verify() {
    // y'' - 3y' + 2y = 6  (distinct real roots r=1,2)
    // Particular: y_p = 6/2 = 3
    // Homogeneous: y_h = C1*exp(x) + C2*exp(2x)
    // General: y = C1*exp(x) + C2*exp(2x) + 3
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let six = symplex::default_context().int(6);
    let ode = &d2y - &(&dy * 3) + &(&y * 2) - &six; // y'' - 3y' + 2y - 6 = 0

    let (sol, constants) = ode
        .solve_ode(&y, &x)
        .expect("y'' - 3y' + 2y = 6 should be solvable");

    // Verify at multiple points for robustness
    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 1, 4);
    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 1, 2);
    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 3, 4);

    // Also verify with C1=0, C2=0 to isolate the particular solution
    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    let zero = symplex::default_context().int(0);
    let particular = sol.subs(&c1, &zero).subs(&c2, &zero);
    let particular_s = format!("{particular}");
    // The particular solution should simplify to 3
    let particular_val = particular.eval_f64();
    if let Ok(v) = particular_val {
        assert!(
            (v - 3.0).abs() < 1e-10,
            "particular solution should be 3, got {v} ({particular_s})"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: Homogeneous still works (regression)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_homogeneous_still_works() {
    // y'' + y = 0 should still be solved by the homogeneous path
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &y; // y'' + y = 0

    let result = ode.solve_ode(&y, &x);
    // This should still solve (complex roots ±i)
    if let Some((sol, constants)) = result {
        let s = format!("{sol}");
        assert!(
            s.contains("C1") && s.contains("C2"),
            "homogeneous should have two constants: {s}"
        );
        assert_eq!(constants.len(), 2);
    }

    // Another homogeneous: y'' - 3y' + 2y = 0
    let ode2 = &d2y - &(&dy * 3) + &(&y * 2);
    let (sol2, constants2) = ode2
        .solve_ode(&y, &x)
        .expect("y'' - 3y' + 2y = 0 should still be solvable");
    let s2 = format!("{sol2}");
    assert!(
        s2.contains("C1") && s2.contains("C2"),
        "homogeneous should have two constants: {s2}"
    );
    assert_eq!(constants2.len(), 2);
    verify_second_order_numerically(&ode2, &sol2, &constants2, &y, &x, 3, 10);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: classify_ode returns correct type for nonhomogeneous
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_classify_nonhomogeneous() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);

    // y'' + y - 1 = 0 should be classified as nonhomogeneous
    let one = symplex::default_context().int(1);
    let ode = &d2y + &y - &one;
    assert_eq!(
        ode.classify_ode(&y, &x),
        OdeType::SecondOrderLinearCCNonHomogeneous,
        "y'' + y = 1 should classify as nonhomogeneous"
    );

    // y'' - 3y' + 2y - x = 0 should be classified as nonhomogeneous
    let ode2 = &d2y - &(&dy * 3) + &(&y * 2) - &x;
    assert_eq!(
        ode2.classify_ode(&y, &x),
        OdeType::SecondOrderLinearCCNonHomogeneous,
        "y'' - 3y' + 2y = x should classify as nonhomogeneous"
    );

    // y'' + y = 0 should still classify as homogeneous
    let ode3 = &d2y + &y;
    assert_eq!(
        ode3.classify_ode(&y, &x),
        OdeType::SecondOrderLinearCCHomogeneous,
        "y'' + y = 0 should classify as homogeneous"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 7: Non-polynomial forcing returns None gracefully
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_exponential_rhs() {
    // y'' + y = exp(x) — exponential forcing is NOT a polynomial,
    // so undetermined coefficients for polynomial rhs should not apply.
    // The solver should return None (or possibly handle it in the future).
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &y - &x.exp(); // y'' + y - exp(x) = 0

    let result = ode.solve_ode(&y, &x);
    // It's OK if this returns None — we just must not return a wrong answer.
    // If it does return something, verify it numerically to ensure correctness.
    if let Some((sol, constants)) = result {
        verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 1, 2);
    }
    // No assertion failure = pass (graceful None or correct answer)
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 8: First-order still works (regression)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_first_order_still_works() {
    // Regression: y' + y = 0 still solves correctly
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let ode = &dy + &y; // y' + y = 0

    let (sol, constants) = ode
        .solve_ode(&y, &x)
        .expect("y' + y = 0 should still be solvable");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have constant: {s}");
    assert!(s.contains("exp"), "should contain exp: {s}");
    assert_eq!(constants.len(), 1, "first-order should have one constant");

    verify_first_order_numerically(&ode, &sol, &constants, &y, &x, 1, 2);

    // Also: y' = x should still work (simple separable)
    let ode2 = &dy - &x; // y' - x = 0
    let (sol2, constants2) = ode2
        .solve_ode(&y, &x)
        .expect("y' = x should still be solvable");
    let s2 = format!("{sol2}");
    assert!(s2.contains("C1"), "should have constant: {s2}");
    assert_eq!(constants2.len(), 1);

    verify_first_order_numerically(&ode2, &sol2, &constants2, &y, &x, 3, 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional: distinct real roots with linear forcing
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_distinct_roots_linear_forcing() {
    // y'' - 3y' + 2y = x  (roots r=1, r=2)
    // Particular: y_p = Ax + B
    //   0 - 3A + 2(Ax + B) = x  ⟹  2A = 1 → A=1/2, -3A + 2B = 0 → B=3/4
    //   y_p = x/2 + 3/4
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y - &(&dy * 3) + &(&y * 2) - &x; // y'' - 3y' + 2y - x = 0

    let (sol, constants) = ode
        .solve_ode(&y, &x)
        .expect("y'' - 3y' + 2y = x should be solvable");

    let s = format!("{sol}");
    assert!(
        s.contains("C1") && s.contains("C2"),
        "should have two constants: {s}"
    );

    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 1, 4);
    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 1, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional: repeated root with constant forcing
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_repeated_root_constant_forcing() {
    // y'' - 2y' + y = 4  (repeated root r=1)
    // Particular: y_p = 4/1 = 4  (since c=1)
    // Homogeneous: y_h = (C1 + C2*x)*exp(x)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let four = symplex::default_context().int(4);
    let ode = &d2y - &(&dy * 2) + &y - &four; // y'' - 2y' + y - 4 = 0

    let (sol, constants) = ode
        .solve_ode(&y, &x)
        .expect("y'' - 2y' + y = 4 should be solvable");

    let s = format!("{sol}");
    assert!(
        s.contains("C1") && s.contains("C2"),
        "should have two constants: {s}"
    );

    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 1, 5);
    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 3, 10);
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional: c = 0 case (y'' + by' = f(x))
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_c_zero_constant_forcing() {
    // y'' + y' - 2 = 0  ⟹  y'' + y' = 2
    // c=0, b=1, rhs = constant 2
    // Case 2: y_p = B_0*x → y_p' = B_0, y_p'' = 0 → b*B_0 = 2 → B_0 = 2
    // y_p = 2x
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let two = symplex::default_context().int(2);
    let ode = &d2y + &dy - &two; // y'' + y' - 2 = 0

    let result = ode.solve_ode(&y, &x);
    assert!(result.is_some(), "y'' + y' = 2 should be solvable");

    let (sol, constants) = result.unwrap();
    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 1, 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional: c = 0, b = 0 case (y'' = f(x))
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_b_c_zero_constant_forcing() {
    // y'' - 6 = 0  ⟹  y'' = 6
    // c=0, b=0, rhs = 6
    // Case 3: y_p = 6/(1*2) * x² = 3x²
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let six = symplex::default_context().int(6);
    let ode = &d2y - &six; // y'' - 6 = 0

    let result = ode.solve_ode(&y, &x);
    assert!(result.is_some(), "y'' = 6 should be solvable");

    let (sol, constants) = result.unwrap();
    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 1, 2);
    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 2, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional: checkodesol integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_nonhomogeneous_checkodesol() {
    // y'' - 3y' + 2y = 6 → particular y_p = 3
    // Check that a known particular solution passes checkodesol
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let six = symplex::default_context().int(6);
    let ode = &d2y - &(&dy * 3) + &(&y * 2) - &six;

    // y = 3 should be a particular solution
    let particular = symplex::default_context().int(3);
    let ok = ode.check_ode_solution(&particular, &y, &x);
    assert!(ok, "y=3 should satisfy y'' - 3y' + 2y = 6");

    // y = 0 should NOT be a solution
    let wrong = symplex::default_context().int(0);
    let not_ok = ode.check_ode_solution(&wrong, &y, &x);
    assert!(!not_ok, "y=0 should NOT satisfy y'' - 3y' + 2y = 6");
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional: scaled leading coefficient
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_scaled_leading_coefficient() {
    // 2y'' + 2y = 4  ⟹  y'' + y = 2  ⟹  y_p = 2
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let two = symplex::default_context().int(2);
    let four = symplex::default_context().int(4);
    let ode = &(&d2y * 2) + &(&y * 2) - &four; // 2y'' + 2y - 4 = 0

    let result = ode.solve_ode(&y, &x);
    assert!(
        result.is_some(),
        "2y'' + 2y = 4 should be solvable (normalises to y'' + y = 2)"
    );

    let (sol, constants) = result.unwrap();
    let _ = two; // suppress unused warning
    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 1, 3);
}
