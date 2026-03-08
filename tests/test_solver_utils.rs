//! Tests for solver utility functions: check_solution, classify_ode, checkodesol.

use symplex::ode::OdeType;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// check_solution — verify algebraic solutions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn checksol_quadratic_root() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; x);
    let poly = expr!(x ^ 2 - 4);
    assert!(poly.check_solution(&x, &symplex::default_context().int(2)));
    assert!(poly.check_solution(&x, &symplex::default_context().int(-2)));
    assert!(!poly.check_solution(&x, &symplex::default_context().int(3)));
}

#[test]
fn checksol_linear() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; x);
    // 3x - 9 = 0  →  x = 3
    let eq = &(&x * 3) - 9;
    assert!(eq.check_solution(&x, &symplex::default_context().int(3)));
    assert!(!eq.check_solution(&x, &symplex::default_context().int(0)));
}

#[test]
fn checksol_cubic_root() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; x);
    let poly = expr!(x ^ 3 - 8);
    assert!(poly.check_solution(&x, &symplex::default_context().int(2)));
    assert!(!poly.check_solution(&x, &symplex::default_context().int(-2)));
}

#[test]
fn checksol_with_trig() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; x);
    let eq = x.sin();
    assert!(eq.check_solution(&x, &symplex::default_context().int(0)));
    assert!(eq.check_solution(&x, &symplex::default_context().pi()));
}

#[test]
fn solve_then_check() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; x);
    let poly = expr!(x ^ 2 - 5 * x + 6);
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty(), "solver should find roots of x²-5x+6");
    for root in &roots {
        assert!(
            poly.check_solution(&x, root),
            "root {root} should satisfy x²-5x+6=0"
        );
    }
}

#[test]
fn checksol_zero_is_root_of_x() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; x);
    assert!(x.check_solution(&x, &symplex::default_context().int(0)));
    assert!(!x.check_solution(&x, &symplex::default_context().int(1)));
}

// ═══════════════════════════════════════════════════════════════════════════
// classify_ode — ODE classification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn classify_simple_separable() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    // y' - x = 0  →  y' = x  (no y dependence → SimpleSeparable)
    let ode = &dy - &x;
    assert_eq!(ode.classify_ode(&y, &x), OdeType::SimpleSeparable);
}

#[test]
fn classify_first_order_linear_cc() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    // y' + 2*y = 0  →  first-order linear CC
    let ode = &dy + &(&y * 2);
    assert_eq!(ode.classify_ode(&y, &x), OdeType::FirstOrderLinearCC);
}

#[test]
fn classify_second_order_homogeneous() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    // y'' + y = 0
    let ode = &d2y + &y;
    assert_eq!(
        ode.classify_ode(&y, &x),
        OdeType::SecondOrderLinearCCHomogeneous
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// checkodesol — verify ODE solutions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn checkodesol_simple_separable() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    // ODE: y' - x = 0  →  solution y = x²/2
    let ode = &dy - &x;
    let sol = &x.powi(2) / 2;
    assert!(ode.check_ode_solution(&sol, &y, &x));
}

#[test]
fn checkodesol_wrong_solution_rejected() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    // ODE: y' - x = 0  →  y = x is NOT a solution (y' = 1 ≠ x in general)
    let ode = &dy - &x;
    assert!(!ode.check_ode_solution(&x, &y, &x));
}

#[test]
fn dsolve_then_checkodesol() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    // y' - x = 0
    let ode = &dy - &x;
    let (sol, _constants) = ode.solve_ode(&y, &x).expect("dsolve should solve y' - x = 0");
    // Substitute C1 = 0 to get a particular solution
    let c1 = symplex::default_context().symbol("C1");
    let particular = sol.subs(&c1, &symplex::default_context().int(0));
    assert!(
        ode.check_ode_solution(&particular, &y, &x),
        "dsolve solution (with C1=0) should satisfy the ODE"
    );
}
