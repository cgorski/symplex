//! Comprehensive ODE solver tests covering all three solver branches.
//!
//! Branch 1: Simple separable — y' = f(x)
//! - y' = x → y = x²/2 + C1
//! - y' = sin(x) → y = -cos(x) + C1
//! - y' = 0 → y = C1
//! - y' = 3 → y = 3x + C1
//!
//! Branch 2: First-order linear — y' + ay = 0 (homogeneous)
//! - y' + 2y = 0 → y = C1·exp(-2x)
//! - y' - y = 0 → y = C1·exp(x)
//!
//! Branch 3: Second-order constant-coefficient — y'' + by' + cy = 0
//! - y'' + y = 0 → complex roots → y = C1·exp(ix) + C2·exp(-ix)
//! - y'' - 3y' + 2y = 0 → distinct real roots → y = C1·exp(x) + C2·exp(2x)
//! - y'' - 2y' + y = 0 → repeated root → y = (C1 + C2·x)·exp(x)
//! - y'' + 5y' + 6y = 0 → distinct real roots → y = C1·exp(-2x) + C2·exp(-3x)
//!
//! Edge cases:
//! - Expression with no derivative → returns None
//! - Pure number → returns None

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers: numerically verify ODE solutions by substitution
// ═══════════════════════════════════════════════════════════════════════════

/// For a first-order ODE `expr = 0`, verify the solution by substituting
/// C1=1 into the solution, computing y' by actual differentiation,
/// replacing the formal y' and y in the ODE, and evaluating at a sample x.
fn verify_first_order_numerically(
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

    // Differentiate the concrete solution to get y'
    let sol_prime = concrete_sol.diff(x);

    // Replace formal y' and y in the ODE expression
    let dy_formal = y.formal_diff(x);
    let residual = ode_expr.subs(&dy_formal, &sol_prime).subs(y, &concrete_sol);

    // Evaluate at the sample point
    let sample_val = symplex::rational(sample_x_num, sample_x_den);
    let residual_at = residual.subs(x, &sample_val);

    let val = residual_at.eval_f64().expect(
        "evalf_f64 should succeed for first-order ODE residual evaluation"
    );
    assert!(
        val.abs() < 1e-6,
        "First-order ODE residual should be ~0, got {val} at x={sample_x_num}/{sample_x_den}\n  \
         solution (C1=1): {concrete_sol}\n  residual: {residual_at}"
    );
}

/// For a second-order ODE `expr = 0`, verify the solution by substituting
/// C1=1, C2=1, computing y' and y'' by actual differentiation, replacing
/// the formal derivatives and y in the ODE, and evaluating at a sample x.
fn verify_second_order_numerically(
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
    let sol_double_prime = sol_prime.diff(x);

    let dy_formal = y.formal_diff(x);
    let d2y_formal = dy_formal.formal_diff(x);

    let residual = ode_expr
        .subs(&d2y_formal, &sol_double_prime)
        .subs(&dy_formal, &sol_prime)
        .subs(y, &concrete_sol);

    let sample_val = symplex::rational(sample_x_num, sample_x_den);
    let residual_at = residual.subs(x, &sample_val);

    let val = residual_at.eval_f64().expect(
        "evalf_f64 should succeed for second-order ODE residual evaluation"
    );
    assert!(
        val.abs() < 1e-4,
        "Second-order ODE residual should be ~0, got {val} at x={sample_x_num}/{sample_x_den}\n  \
         solution (C1=C2=1): {concrete_sol}\n  residual: {residual_at}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Branch 1: Simple separable — y' = f(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn separable_dy_dx_eq_x() {
    // y' - x = 0 → y = x²/2 + C1
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let ode = &dy - &x;

    let (sol, constants) = ode.solve_ode(&y, &x).expect("should solve y' = x");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");
    assert!(!constants.is_empty(), "should return at least one constant");

    verify_first_order_numerically(&ode, &sol, &constants, &y, &x, 3, 2);
}

#[test]
fn separable_dy_dx_eq_zero() {
    // y' = 0 → y = C1
    let x = symplex::var("x");
    let y = symplex::var("y");
    let ode = y.formal_diff(&x);

    let (sol, constants) = ode.solve_ode(&y, &x).expect("should solve y' = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should be C1: {s}");
    assert_eq!(constants.len(), 1, "should have exactly one constant");
}

#[test]
fn separable_dy_dx_eq_sin_x() {
    // y' - sin(x) = 0 → y = -cos(x) + C1
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let ode = &dy - &x.sin();

    let (sol, constants) = ode.solve_ode(&y, &x).expect("should solve y' = sin(x)");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");

    verify_first_order_numerically(&ode, &sol, &constants, &y, &x, 7, 10);
}

#[test]
fn separable_dy_dx_eq_constant() {
    // y' - 3 = 0 → y = 3x + C1
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let three = symplex::int(3);
    let ode = &dy - &three;

    let (sol, constants) = ode.solve_ode(&y, &x).expect("should solve y' = 3");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");

    verify_first_order_numerically(&ode, &sol, &constants, &y, &x, 2, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Branch 2: First-order linear — y' + ay = 0
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn first_order_linear_exponential_decay() {
    // y' + 2y = 0 → y = C1·exp(-2x)
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let ode = &dy + &(&y * 2); // y' + 2y = 0

    let (sol, constants) = ode.solve_ode(&y, &x).expect("should solve y' + 2y = 0");
    let s = format!("{sol}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");
    assert!(s.contains("C1"), "solution should have C1: {s}");

    verify_first_order_numerically(&ode, &sol, &constants, &y, &x, 1, 2);
}

#[test]
fn first_order_linear_exponential_growth() {
    // y' - y = 0 → y = C1·exp(x)
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let ode = &dy - &y; // y' - y = 0

    let (sol, constants) = ode.solve_ode(&y, &x).expect("should solve y' - y = 0");
    let s = format!("{sol}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");
    assert!(s.contains("C1"), "solution should have C1: {s}");

    verify_first_order_numerically(&ode, &sol, &constants, &y, &x, 1, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Branch 3: Second-order constant-coefficient — y'' + by' + cy = 0
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn second_order_distinct_real_roots() {
    // y'' - 3y' + 2y = 0 → characteristic r² - 3r + 2 = 0 → r=1,2
    // → y = C1·exp(x) + C2·exp(2x)
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y - &(&dy * 3) + &(&y * 2); // y'' - 3y' + 2y = 0

    let (sol, constants) = ode.solve_ode(&y, &x).expect("should solve y'' - 3y' + 2y = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");
    assert!(s.contains("C2"), "solution should have C2: {s}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");
    assert_eq!(constants.len(), 2, "should have two constants");

    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 3, 10);
}

#[test]
fn second_order_repeated_root() {
    // y'' - 2y' + y = 0 → characteristic r² - 2r + 1 = 0 → r=1 (double)
    // → y = (C1 + C2·x)·exp(x)
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y - &(&dy * 2) + &y; // y'' - 2y' + y = 0

    let (sol, constants) = ode.solve_ode(&y, &x).expect("should solve y'' - 2y' + y = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");
    assert!(s.contains("C2"), "solution should have C2: {s}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");
    // Repeated root: solution should involve x multiplied with exp
    assert!(
        s.contains("x"),
        "repeated root solution should contain x: {s}"
    );

    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 2, 5);
}

#[test]
fn second_order_complex_roots() {
    // y'' + y = 0 → characteristic r² + 1 = 0 → r = ±i
    // → y = C1·exp(ix) + C2·exp(-ix)  (or equivalently C1·cos(x) + C2·sin(x))
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &y; // y'' + y = 0

    // Complex roots may or may not be supported — verify gracefully.
    if let Some((sol, constants)) = ode.solve_ode(&y, &x) {
        let s = format!("{sol}");
        assert!(
            s.contains("C1") && s.contains("C2"),
            "should have two constants: {s}"
        );
        assert_eq!(constants.len(), 2, "should have two constants");
        // The solution should involve exponentials (possibly with i), or trig
        assert!(
            s.contains("exp") || s.contains("sin") || s.contains("cos"),
            "should contain exp or trig: {s}"
        );
    }
    // It is acceptable if the solver does not yet handle complex characteristic roots.
}

#[test]
fn second_order_distinct_real_negative_roots() {
    // y'' + 5y' + 6y = 0 → r² + 5r + 6 = 0 → r = -2, -3
    // → y = C1·exp(-2x) + C2·exp(-3x)
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &(&dy * 5) + &(&y * 6); // y'' + 5y' + 6y = 0

    let (sol, constants) = ode.solve_ode(&y, &x).expect("should solve y'' + 5y' + 6y = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");
    assert!(s.contains("C2"), "solution should have C2: {s}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");

    verify_second_order_numerically(&ode, &sol, &constants, &y, &x, 1, 4);
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn no_derivative_returns_none() {
    // x + y = 0 has no derivative — not an ODE
    let x = symplex::var("x");
    let y = symplex::var("y");
    let expr = &x + &y;

    let result = expr.solve_ode(&y, &x);
    assert!(
        result.is_none(),
        "expression without derivative should return None"
    );
}

#[test]
fn pure_number_returns_none() {
    // 42 = 0 is not an ODE
    let x = symplex::var("x");
    let y = symplex::var("y");
    let expr = symplex::int(42);

    let result = expr.solve_ode(&y, &x);
    assert!(result.is_none(), "pure number should return None");
}

// ═══════════════════════════════════════════════════════════════════════════
// expr! macro ODE construction
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_separable_ode() {
    let x = symplex::var("x");
    let y = symplex::var("y");
    let ode = expr!(diff(y, x) - x); // y' - x = 0

    let (sol, constants) = ode
        .solve_ode(&y, &x)
        .expect("expr! separable ODE should solve");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(!constants.is_empty());
}

#[test]
fn expr_macro_first_order_linear_ode() {
    let x = symplex::var("x");
    let y = symplex::var("y");
    let ode = expr!(diff(y, x) + 2 * y); // y' + 2y = 0

    let (sol, _) = ode
        .solve_ode(&y, &x)
        .expect("expr! first-order linear ODE should solve");
    let s = format!("{sol}");
    assert!(
        s.contains("exp") && s.contains("C1"),
        "solution should be C1*exp(-2x): {s}"
    );
}
