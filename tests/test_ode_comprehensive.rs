//! Comprehensive ODE solver tests covering **all** ODE solver classes.
//!
//! Each test targets a specific `OdeType` variant and verifies:
//! 1. The solver returns `Some(...)` (or gracefully returns `None` for harder classes).
//! 2. The solution contains the expected arbitrary constants.
//! 3. Numerical substitution of the solution back into the ODE yields a residual ≈ 0.
//!
//! ODE classes covered:
//!  1. SimpleSeparable         — y' = f(x)
//!  2. FullSeparable           — y' = f(x)·g(y)
//!  3. FirstOrderLinearCC      — y' + a·y = f(x)  (constant coeff)
//!  4. FirstOrderLinearVC      — y' + P(x)·y = Q(x) (variable coeff)
//!  5. ExactFirstOrder         — M(x,y) + N(x,y)·y' = 0
//!  6. Bernoulli               — y' + P(x)·y = Q(x)·yⁿ
//!  7. SecondOrderLinearCCHomogeneous (distinct real roots)
//!  8. SecondOrderLinearCCHomogeneous (repeated root)
//!  9. SecondOrderLinearCCHomogeneous (complex roots)
//! 10. SecondOrderLinearCCNonHomogeneous — y'' + by' + cy = f(x)
//! 11. EulerCauchy             — a·x²y'' + b·xy' + c·y = 0
//! 12. VariationOfParameters   — y'' + py' + qy = g(x)
//! 13. HomogeneousCoefficient  — y' = f(y/x)
//! 14. NthOrderReducible       — F(y, y', y'') = 0 (no explicit x)

use symplex::ode::OdeType;
use symplex::prelude::*;
use symplex::expr::ExprType;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers: numerically verify ODE solutions by substitution
// ═══════════════════════════════════════════════════════════════════════════

/// For a first-order ODE `expr = 0`, verify the solution by substituting
/// all constants to 1, computing y' by differentiation, replacing the
/// formal y' and y in the ODE, and evaluating at sample points.
fn verify_first_order_numerically(
    ode_expr: &Ex,
    solution: &Ex,
    constants: &[Ex],
    y: &Ex,
    x: &Ex,
    points: &[(i64, i64)],
) {
    let one = symplex::default_context().int(1);
    let mut concrete_sol = solution.clone();
    for c in constants {
        concrete_sol = concrete_sol.subs(c, &one);
    }

    let sol_prime = concrete_sol.diff(x);
    let dy_formal = y.formal_diff(x);
    let residual = ode_expr.subs(&dy_formal, &sol_prime).subs(y, &concrete_sol);

    let mut checked = 0usize;
    for &(num, den) in points {
        let sample_val = symplex::default_context().rational(num, den);
        let residual_at = residual.subs(x, &sample_val);

        if let Ok(val) = residual_at.eval_f64() {
            if val.is_finite() {
                checked += 1;
                assert!(
                    val.abs() < 1e-4,
                    "First-order ODE residual should be ~0, got {val} at x={num}/{den}\n  \
                     solution (C=1): {concrete_sol}\n  residual: {residual_at}"
                );
            }
        }
    }
    assert!(
        checked > 0,
        "First-order ODE: no points could be evaluated — test is vacuous\n  \
         solution (C=1): {concrete_sol}"
    );
}

/// For a second-order ODE `expr = 0`, verify the solution by substituting
/// all constants to 1, computing y' and y'' by differentiation, replacing
/// formal derivatives and y in the ODE, and evaluating at sample points.
fn verify_second_order_numerically(
    ode_expr: &Ex,
    solution: &Ex,
    constants: &[Ex],
    y: &Ex,
    x: &Ex,
    points: &[(i64, i64)],
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

    let mut checked = 0usize;
    for &(num, den) in points {
        let sample_val = symplex::default_context().rational(num, den);
        let residual_at = residual.subs(x, &sample_val);

        if let Ok(val) = residual_at.eval_f64() {
            if val.is_finite() {
                checked += 1;
                assert!(
                    val.abs() < 1e-3,
                    "Second-order ODE residual should be ~0, got {val} at x={num}/{den}\n  \
                     solution (C=1): {concrete_sol}\n  residual: {residual_at}"
                );
            }
        }
    }
    assert!(
        checked > 0,
        "Second-order ODE: no points could be evaluated — test is vacuous\n  \
         solution (C=1): {concrete_sol}"
    );
}

/// Standard first-order sample points (positive rationals to avoid poles).
const FIRST_ORDER_POINTS: &[(i64, i64)] = &[(3, 2), (7, 10), (2, 1), (11, 4)];

/// Standard second-order sample points.
const SECOND_ORDER_POINTS: &[(i64, i64)] = &[(3, 10), (1, 2), (7, 10), (2, 1)];

/// Points that are positive (for ln, sqrt, etc.).
const POSITIVE_POINTS: &[(i64, i64)] = &[(1, 2), (3, 2), (2, 1), (5, 2)];

// ═══════════════════════════════════════════════════════════════════════════
// 1. SimpleSeparable: y' = f(x) — no y dependence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_simple_separable_x_squared() {
    // y' = x² → y = x³/3 + C1
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let x_sq = x.powi(2);
    let ode = &dy - &x_sq; // y' - x² = 0

    // Classification
    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::SimpleSeparable,
        "y' = x² should classify as SimpleSeparable, got {ode_type:?}"
    );

    // Solve
    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y' = x²");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");

    // Verify numerically
    let c1 = symplex::default_context().symbol("C1");
    verify_first_order_numerically(&ode, &sol, &[c1], &y, &x, FIRST_ORDER_POINTS);
}

#[test]
fn comprehensive_simple_separable_expr_macro() {
    // y' - x² = 0 via expr! macro
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let ode = expr!(diff(y, x) - x ^ 2);

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("expr! simple separable should solve");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");

    let c1 = symplex::default_context().symbol("C1");
    verify_first_order_numerically(&ode, &sol, &[c1], &y, &x, FIRST_ORDER_POINTS);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. FullSeparable: y' = f(x)·g(y)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_full_separable_xy() {
    // y' - xy = 0 → y' = xy → separable: dy/y = x dx → ln|y| = x²/2 + C
    // → y = C1·exp(x²/2)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let ode = expr!(diff(y, x) - x * y); // y' - xy = 0

    // Classification
    let ode_type = ode.classify_ode(&y, &x);
    // FullSeparable or FirstOrderLinearVC or FirstOrderLinearCC are all plausible
    // dispatches depending on the classifier priority. We just verify it solves.
    eprintln!("y' - xy = 0 classified as: {ode_type:?}");

    // Solve
    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y' = xy");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");
    assert!(
        s.contains("exp"),
        "solution should involve exp: {s}"
    );

    // Verify numerically
    let c1 = symplex::default_context().symbol("C1");
    verify_first_order_numerically(&ode, &sol, &[c1], &y, &x, FIRST_ORDER_POINTS);
}

#[test]
fn comprehensive_full_separable_y_over_x() {
    // y' = y/x → dy/y = dx/x → ln|y| = ln|x| + C → y = C1·x
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let ode = &dy - &(&y / &x); // y' - y/x = 0

    let sol = ode.solve_ode(&y, &x);
    if !sol.has_unevaluated() {
        let s = format!("{sol}");
        assert!(
            s.contains("C1"),
            "solution should have a constant: {s}"
        );
        let c1 = symplex::default_context().symbol("C1");
        verify_first_order_numerically(&ode, &sol, &[c1], &y, &x, POSITIVE_POINTS);
    } else {
        eprintln!("NOTE: y' = y/x not solved — may need exp(ln(x)) simplification");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. FirstOrderLinearCC: y' + a·y = f(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_first_order_linear_cc_homogeneous() {
    // y' + 2y = 0 → y = C1·exp(-2x)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let ode = expr!(diff(y, x) + 2 * y); // y' + 2y = 0

    // Classification
    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::FirstOrderLinearCC,
        "y' + 2y = 0 should classify as FirstOrderLinearCC, got {ode_type:?}"
    );

    // Solve
    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y' + 2y = 0");
    let s = format!("{sol}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");
    assert!(s.contains("C1"), "solution should have C1: {s}");

    // Verify numerically
    let c1 = symplex::default_context().symbol("C1");
    verify_first_order_numerically(&ode, &sol, &[c1], &y, &x, FIRST_ORDER_POINTS);
}

#[test]
fn comprehensive_first_order_linear_cc_nonhomogeneous() {
    // y' + 2y - exp(-x) = 0 → y' + 2y = exp(-x)
    // Integrating factor μ = exp(2x), solution involves exp terms
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let two_y = &y * 2;
    let neg_x = (&x * -1).exp();
    let ode = &(&dy + &two_y) - &neg_x; // y' + 2y - exp(-x) = 0

    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "should solve y' + 2y = exp(-x)");

    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");

    let c1 = symplex::default_context().symbol("C1");
    verify_first_order_numerically(&ode, &sol, &[c1], &y, &x, FIRST_ORDER_POINTS);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. FirstOrderLinearVC: y' + P(x)·y = Q(x) — variable coefficients
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_first_order_linear_vc_homogeneous() {
    // y' + 2xy = 0 → y = C1·exp(-x²)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let ode = expr!(diff(y, x) + 2 * x * y);

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y' + 2xy = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");

    let c1 = symplex::default_context().symbol("C1");
    verify_first_order_numerically(&ode, &sol, &[c1], &y, &x, FIRST_ORDER_POINTS);
}

#[test]
fn comprehensive_first_order_linear_vc_nonhomogeneous() {
    // y' + y/x - x = 0 → y' + (1/x)·y = x
    // Integrating factor μ = exp(∫1/x dx) = x
    // Solution: y = x²/3 + C1/x
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let ode = &dy + &(&y / &x) - &x; // y' + y/x - x = 0

    let sol = ode.solve_ode(&y, &x);
    if !sol.has_unevaluated() {
        let s = format!("{sol}");
        eprintln!("y' + y/x = x  solution: {s}");
        assert!(
            s.contains("C1"),
            "should have constant C1: {s}"
        );
        // Verify at positive x (to avoid singularity at x=0)
        let c1 = symplex::default_context().symbol("C1");
        verify_first_order_numerically(&ode, &sol, &[c1], &y, &x, POSITIVE_POINTS);
    } else {
        eprintln!(
            "NOTE: y' + y/x = x not yet solved — may need exp(ln(x)) simplification"
        );
    }
}

#[test]
fn comprehensive_first_order_linear_vc_3x_squared() {
    // y' + 3x²y = 0 → P(x) = 3x², ∫P dx = x³ → y = C1·exp(-x³)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let three = symplex::default_context().int(3);
    let x_sq = x.powi(2);
    let ode = &dy + &(&three * &x_sq * &y);

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y' + 3x²y = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");

    let c1 = symplex::default_context().symbol("C1");
    verify_first_order_numerically(&ode, &sol, &[c1], &y, &x, FIRST_ORDER_POINTS);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. ExactFirstOrder: M(x,y) + N(x,y)·y' = 0 with ∂M/∂y = ∂N/∂x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_exact_first_order() {
    // (2xy + 3) + (x² + 4y)·y' = 0
    // M = 2xy + 3, N = x² + 4y
    // ∂M/∂y = 2x, ∂N/∂x = 2x — exact!
    // F = ∫M dx = x²y + 3x + g(y), g'(y) = 4y → g = 2y²
    // Solution: x²y + 3x + 2y² = C1
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);

    let two = symplex::default_context().int(2);
    let three = symplex::default_context().int(3);
    let four = symplex::default_context().int(4);
    let m = &(&two * &x * &y) + &three; // 2xy + 3
    let n = &x.powi(2) + &(&four * &y); // x² + 4y
    let ode = &m + &(&n * &dy);

    // Classification
    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::ExactFirstOrder,
        "(2xy+3) + (x²+4y)y' = 0 should classify as ExactFirstOrder, got {ode_type:?}"
    );

    // Solve
    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");
    // Exact ODE solver returns potential F(x,y); constant is implicit.
    assert!(
        sol.expr_type() != ExprType::Unevaluated,
        "exact ODE (2xy+3) + (x²+4y)y' = 0 should be solvable, got: {s}"
    );
    assert!(!s.is_empty(), "solution should be non-empty: {s}");
}

#[test]
fn comprehensive_exact_simple_ydx_xdy() {
    // y + x·y' = 0 → M = y, N = x → ∂M/∂y = 1, ∂N/∂x = 1 — exact
    // F = xy, solution: xy = C1
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let ode = &y + &(&x * &dy); // y + x·y' = 0

    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "y + x·y' = 0 should be solvable (exact)");

    let s = format!("{sol}");
    assert!(
        s.contains("C1"),
        "solution should have a constant: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Bernoulli: y' + P(x)·y = Q(x)·yⁿ  (n ≠ 0, 1)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_bernoulli_n2_constant_coeff() {
    // y' + y - y² = 0  (P=1, Q=1, n=2)
    // Substitution v = 1/y → v' - v = -1
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let y_sq = y.powi(2);
    let ode = &(&dy + &y) - &y_sq;

    // Classification
    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::Bernoulli,
        "y' + y - y² = 0 should classify as Bernoulli, got {ode_type:?}"
    );

    // Solve
    let sol = ode.solve_ode(&y, &x);
    assert!(
        !sol.has_unevaluated(),
        "Bernoulli y' + y - y² = 0 should be solvable"
    );

    let s = format!("{sol}");
    assert!(
        s.contains("C1"),
        "solution should have a constant: {s}"
    );

    // Verify numerically
    let c1 = symplex::default_context().symbol("C1");
    verify_first_order_numerically(&ode, &sol, &[c1], &y, &x, FIRST_ORDER_POINTS);
}

#[test]
fn comprehensive_bernoulli_y_over_x() {
    // y' + y/x = y²  →  y' + (1/x)·y − y² = 0
    // Bernoulli with P(x) = 1/x, Q(x) = 1, n = 2
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let y_over_x = &y / &x;
    let y_sq = y.powi(2);
    let ode = &(&dy + &y_over_x) - &y_sq;

    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::Bernoulli,
        "y' + y/x - y² = 0 should classify as Bernoulli, got {ode_type:?}"
    );

    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");
    // The old API returned Option — is_some() just meant the solver returned
    // *something*, even if it contained unevaluated sub-expressions.
    assert!(
        sol.expr_type() != ExprType::Unevaluated,
        "Bernoulli ODE y' + y/x = y² should be solvable, got: {s}"
    );
    assert!(!s.is_empty(), "Bernoulli solution should be non-empty: {s}");
}

#[test]
fn comprehensive_bernoulli_n2_p_equals_2() {
    // y' + 2y − y² = 0 (P=2, Q=1, n=2)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let two = symplex::default_context().int(2);
    let y_sq = y.powi(2);
    let ode = &(&dy + &(&two * &y)) - &y_sq;

    let sol = ode.solve_ode(&y, &x);
    assert!(
        !sol.has_unevaluated(),
        "Bernoulli y' + 2y - y² = 0 should be solvable"
    );

    let s = format!("{sol}");
    assert!(
        s.contains("C1"),
        "solution should have a constant: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. SecondOrderLinearCCHomogeneous — distinct real roots
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_second_order_cc_distinct_real() {
    // y'' - 3y' + 2y = 0 → r² - 3r + 2 = 0 → (r-1)(r-2) = 0
    // → y = C1·exp(x) + C2·exp(2x)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y - &(&dy * 3) + &(&y * 2); // y'' - 3y' + 2y = 0

    // Classification
    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::SecondOrderLinearCCHomogeneous,
        "y'' - 3y' + 2y = 0 should be SecondOrderLinearCCHomogeneous, got {ode_type:?}"
    );

    // Solve
    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y'' - 3y' + 2y = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(s.contains("C2"), "should have C2: {s}");
    assert!(s.contains("exp"), "should contain exp: {s}");

    // Verify numerically at multiple points
    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, SECOND_ORDER_POINTS);
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. SecondOrderLinearCCHomogeneous — repeated root
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_second_order_cc_repeated() {
    // y'' - 2y' + y = 0 → r² - 2r + 1 = 0 → (r-1)² = 0 → r = 1 (double)
    // → y = (C1 + C2·x)·exp(x)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y - &(&dy * 2) + &y; // y'' - 2y' + y = 0

    // Classification
    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::SecondOrderLinearCCHomogeneous,
        "y'' - 2y' + y = 0 should be SecondOrderLinearCCHomogeneous, got {ode_type:?}"
    );

    // Solve
    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y'' - 2y' + y = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(s.contains("C2"), "should have C2: {s}");
    assert!(s.contains("exp"), "should contain exp: {s}");
    // Repeated root: solution involves x * exp(x)
    assert!(
        s.contains("x"),
        "repeated root solution should contain x: {s}"
    );

    // Verify numerically
    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, SECOND_ORDER_POINTS);
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. SecondOrderLinearCCHomogeneous — complex roots
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_second_order_cc_complex() {
    // y'' + y = 0 → r² + 1 = 0 → r = ±i
    // → y = C1·cos(x) + C2·sin(x) or equivalently C1·exp(ix) + C2·exp(-ix)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &y; // y'' + y = 0

    // Classification
    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::SecondOrderLinearCCHomogeneous,
        "y'' + y = 0 should be SecondOrderLinearCCHomogeneous, got {ode_type:?}"
    );

    // Complex roots may or may not be fully supported — verify gracefully.
    let sol = ode.solve_ode(&y, &x);
    if !sol.has_unevaluated() {
        let s = format!("{sol}");
        assert!(
            s.contains("C1") && s.contains("C2"),
            "should have two constants: {s}"
        );
        // Solution should involve exponentials (with i) or trig functions
        assert!(
            s.contains("exp") || s.contains("sin") || s.contains("cos"),
            "should contain exp or trig: {s}"
        );
    }
}

#[test]
fn comprehensive_second_order_cc_negative_roots() {
    // y'' + 5y' + 6y = 0 → r² + 5r + 6 = 0 → (r+2)(r+3) = 0 → r = -2, -3
    // → y = C1·exp(-2x) + C2·exp(-3x)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &(&dy * 5) + &(&y * 6); // y'' + 5y' + 6y = 0

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y'' + 5y' + 6y = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(s.contains("C2"), "should have C2: {s}");
    assert!(s.contains("exp"), "should contain exp: {s}");

    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, SECOND_ORDER_POINTS);
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. SecondOrderLinearCCNonHomogeneous: y'' + by' + cy = f(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_second_order_cc_nonhomogeneous_linear_rhs() {
    // y'' + y = x → y'' + y - x = 0
    // y_h involves exp(±ix), y_p = x (since c=1, try y_p = Ax+B: A=1, B=0)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &y - &x; // y'' + y - x = 0

    // Classification
    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::SecondOrderLinearCCNonHomogeneous,
        "y'' + y = x should classify as SecondOrderLinearCCNonHomogeneous, got {ode_type:?}"
    );

    // Solve
    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "y'' + y = x should be solvable");

    let s = format!("{sol}");
    assert!(
        s.contains("C1") && s.contains("C2"),
        "should have two constants: {s}"
    );

    // Verify numerically
    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, SECOND_ORDER_POINTS);
}

#[test]
fn comprehensive_second_order_cc_nonhomogeneous_constant_rhs() {
    // y'' + y = 1 → y'' + y - 1 = 0
    // y_p = 1 (constant forcing with c=1)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let one = symplex::default_context().int(1);
    let ode = &d2y + &y - &one; // y'' + y - 1 = 0

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y'' + y = 1");
    let s = format!("{sol}");
    assert!(
        s.contains("C1") && s.contains("C2"),
        "should have two constants: {s}"
    );

    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, SECOND_ORDER_POINTS);
}

#[test]
fn comprehensive_second_order_cc_nonhomogeneous_quadratic_rhs() {
    // y'' + y = x²  →  y'' + y - x² = 0
    // y_p = x² - 2 (via undetermined coefficients)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let ode = &d2y + &y - &x_sq; // y'' + y - x² = 0

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y'' + y = x²");
    let s = format!("{sol}");
    assert!(
        s.contains("C1") && s.contains("C2"),
        "should have two constants: {s}"
    );

    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, SECOND_ORDER_POINTS);
}

#[test]
fn comprehensive_second_order_cc_nonhomogeneous_distinct_roots() {
    // y'' - 3y' + 2y = 6 → distinct real roots r=1,2
    // y_p = 6/2 = 3, y_h = C1*exp(x) + C2*exp(2x)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let six = symplex::default_context().int(6);
    let ode = &d2y - &(&dy * 3) + &(&y * 2) - &six; // y'' - 3y' + 2y - 6 = 0

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y'' - 3y' + 2y = 6");

    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(&ode, &sol, &[c1.clone(), c2.clone()], &y, &x, SECOND_ORDER_POINTS);

    // Also verify that with C1=0, C2=0 the particular solution ≈ 3
    let zero = symplex::default_context().int(0);
    let particular = sol.subs(&c1, &zero).subs(&c2, &zero);
    if let Ok(v) = particular.eval_f64() {
        assert!(
            (v - 3.0).abs() < 1e-10,
            "particular solution should be 3, got {v}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. EulerCauchy: a·x²·y'' + b·x·y' + c·y = 0
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_euler_cauchy_distinct_real() {
    // x²y'' + xy' - y = 0
    // Characteristic: a=1, b=1, c=-1
    //   r(r-1) + r - 1 = 0 → r² - 1 = 0 → r = 1, -1
    // → y = C1·x + C2·x⁻¹
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let ode = &(&(&x_sq * &d2y) + &(&x * &dy)) - &y; // x²y'' + xy' - y = 0

    // Classification
    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::EulerCauchy,
        "x²y'' + xy' - y = 0 should classify as EulerCauchy, got {ode_type:?}"
    );

    // Solve
    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve x²y'' + xy' - y = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(s.contains("C2"), "should have C2: {s}");

    // Verify at positive x values (Euler-Cauchy requires x > 0)
    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, POSITIVE_POINTS);
}

#[test]
fn comprehensive_euler_cauchy_x_sq_y_pp_minus_2y() {
    // x²y'' − 2y = 0
    // Characteristic: r(r-1) - 2 = 0 → r² - r - 2 = 0 → (r-2)(r+1) = 0
    // → r = 2, -1 → y = C1·x² + C2·x⁻¹
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let two = symplex::default_context().int(2);
    let ode = &(&x_sq * &d2y) - &(&two * &y);

    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::EulerCauchy,
        "x²y'' - 2y = 0 should classify as EulerCauchy, got {ode_type:?}"
    );

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve x²y'' - 2y = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1") && s.contains("C2"), "should have two constants: {s}");

    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, POSITIVE_POINTS);
}

#[test]
fn comprehensive_euler_cauchy_complex() {
    // x²y'' + xy' + y = 0
    // Characteristic: r(r-1) + r + 1 = 0 → r² + 1 = 0 → r = ±i
    // → y = C1·cos(ln x) + C2·sin(ln x)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let ode = &(&(&x_sq * &d2y) + &(&x * &dy)) + &y;

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve x²y'' + xy' + y = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(s.contains("C2"), "should have C2: {s}");
    assert!(
        s.contains("cos") && s.contains("sin"),
        "complex Euler-Cauchy should use cos and sin: {s}"
    );
    assert!(s.contains("ln"), "should involve ln(x): {s}");

    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, POSITIVE_POINTS);
}

#[test]
fn comprehensive_euler_cauchy_repeated() {
    // x²y'' + xy' = 0
    // Characteristic: r(r-1) + r = 0 → r² = 0 → r = 0 (double)
    // → y = C1 + C2·ln(x)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let ode = &(&x_sq * &d2y) + &(&x * &dy);

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve x²y'' + xy' = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(s.contains("C2"), "should have C2: {s}");
    assert!(
        s.contains("ln"),
        "repeated root Euler-Cauchy should contain ln(x): {s}"
    );

    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, POSITIVE_POINTS);
}

#[test]
fn comprehensive_euler_cauchy_with_coefficients() {
    // 2x²y'' + 3xy' − y = 0
    // 2r(r-1) + 3r - 1 = 0 → 2r² + r - 1 = 0 → (2r-1)(r+1) = 0
    // → r = 1/2, -1 → y = C1·√x + C2/x
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let two = symplex::default_context().int(2);
    let three = symplex::default_context().int(3);
    let ode = &(&(&two * &x_sq * &d2y) + &(&three * &x * &dy)) - &y;

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve 2x²y'' + 3xy' - y = 0");
    let s = format!("{sol}");
    assert!(
        s.contains("C1") && s.contains("C2"),
        "should have two constants: {s}"
    );

    // Verify at x = 4 (nice for √x)
    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(
        &ode,
        &sol,
        &[c1, c2],
        &y,
        &x,
        &[(4, 1), (9, 4), (2, 1)],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. VariationOfParameters: y'' + py' + qy = g(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_variation_of_parameters_tan() {
    // y'' + y = tan(x)
    // Homogeneous: y₁ = cos(x), y₂ = sin(x)
    // This requires VoP since tan(x) is not polynomial.
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &(&d2y + &y) - &x.tan(); // y'' + y - tan(x) = 0

    // The solver should attempt VoP. It may or may not succeed depending on
    // the integration engine's ability to handle ∫sin(x)tan(x) dx.
    let sol = ode.solve_ode(&y, &x);
    if !sol.has_unevaluated() {
        let s = format!("{sol}");
        assert!(
            s.contains("C1") && s.contains("C2"),
            "VoP solution should have two constants: {s}"
        );

        // Verify numerically at a point where tan is well-behaved
        let c1 = symplex::default_context().symbol("C1");
        let c2 = symplex::default_context().symbol("C2");
        verify_second_order_numerically(
            &ode,
            &sol,
            &[c1, c2],
            &y,
            &x,
            &[(1, 4), (1, 10), (3, 10)],
        );
    } else {
        eprintln!("NOTE: y'' + y = tan(x) not solved — VoP integrals may be too hard");
    }
}

#[test]
fn comprehensive_variation_of_parameters_exp() {
    // y'' − y = exp(x) — resonance case, but also solvable by VoP
    // (undetermined coefficients may catch it first)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &(&d2y - &y) - &x.exp(); // y'' - y - exp(x) = 0

    let sol = ode.solve_ode(&y, &x);
    assert!(
        !sol.has_unevaluated(),
        "y'' - y = exp(x) should be solvable (via undetermined coefficients or VoP)"
    );

    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(s.contains("C2"), "should have C2: {s}");

    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, SECOND_ORDER_POINTS);
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. HomogeneousCoefficient: y' = f(y/x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_homogeneous_coefficient_classify() {
    // y' = (x² + y²)/x²  = 1 + (y/x)²
    // After v = y/x substitution: RHS becomes 1 + v² (free of x)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let x_sq = x.powi(2);
    let y_sq = y.powi(2);
    let rhs = &(&x_sq + &y_sq) / &x_sq; // (x² + y²)/x²
    let ode = &dy - &rhs;

    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::HomogeneousCoefficient,
        "y' = (x²+y²)/x² should classify as HomogeneousCoefficient, got {ode_type:?}"
    );

    // Solving: this may or may not succeed — the integration of dv/(1+v²-v) can be tricky.
    let sol = ode.solve_ode(&y, &x);
    if !sol.has_unevaluated() {
        let s = format!("{sol}");
        assert!(
            s.contains("C1"),
            "solution should contain a constant: {s}"
        );
    } else {
        eprintln!("NOTE: HomogeneousCoefficient y' = (x²+y²)/x² not solved");
    }
}

#[test]
fn comprehensive_homogeneous_coefficient_simple() {
    // y' = (x² + y²)/(xy) → RHS can be written as x/y + y/x = 1/v + v
    // where v = y/x
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let x_sq = x.powi(2);
    let y_sq = y.powi(2);
    let rhs = &(&x_sq + &y_sq) / &(&x * &y);
    let ode = &dy - &rhs;

    let sol = ode.solve_ode(&y, &x);
    if !sol.has_unevaluated() {
        let s = format!("{sol}");
        assert!(
            s.contains("C1"),
            "solution should contain a constant: {s}"
        );
    } else {
        eprintln!("NOTE: HomogeneousCoefficient y' = (x²+y²)/(xy) not solved");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. NthOrderReducible: F(y, y', y'') = 0 (no explicit x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_nth_order_reducible_basic() {
    // y'' = y' (missing x explicitly)
    // p = y', dp/dy = 1, p = y + C1, then dy/dx = y + C1 → separable
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y - &dy; // y'' - y' = 0

    // This is also SecondOrderLinearCCHomogeneous, so the classifier may pick
    // that up first. Either way, it should solve.
    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "should solve y'' - y' = 0");

    let s = format!("{sol}");
    assert!(
        s.contains("C1") || s.contains("C2"),
        "solution should contain constants: {s}"
    );

    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, SECOND_ORDER_POINTS);
}

#[test]
fn comprehensive_nth_order_reducible_nonlinear() {
    // y·y'' = (y')² → y·p·dp/dy = p² → y·dp/dy = p → dp/p = dy/y
    // → p = C1·y → dy/dx = C1·y → y = exp(C1·x + C2)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let dy_sq = dy.powi(2);
    let y_d2y = &y * &d2y;
    let ode = &y_d2y - &dy_sq; // y·y'' - (y')² = 0

    let ode_type = ode.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::NthOrderReducible,
        "y·y'' = (y')² should classify as NthOrderReducible, got {ode_type:?}"
    );

    // Nonlinear reducible — may or may not succeed
    let sol = ode.solve_ode(&y, &x);
    if !sol.has_unevaluated() {
        let s = format!("{sol}");
        assert!(
            s.contains("C1") || s.contains("C2"),
            "solution should contain constants: {s}"
        );
    } else {
        eprintln!("NOTE: y·y'' = (y')² not solved (nonlinear reducible)");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-classification: make sure different ODE types are NOT confused
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_not_euler_cauchy() {
    // y'' + y = 0 — second-order CC, NOT Euler-Cauchy (no x² on y'')
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &y;

    let ode_type = ode.classify_ode(&y, &x);
    assert_ne!(
        ode_type,
        OdeType::EulerCauchy,
        "y'' + y = 0 should NOT be Euler-Cauchy"
    );
    assert_eq!(
        ode_type,
        OdeType::SecondOrderLinearCCHomogeneous,
        "y'' + y = 0 should be SecondOrderLinearCCHomogeneous"
    );
}

#[test]
fn comprehensive_not_bernoulli() {
    // y' + y = 0 — linear, NOT Bernoulli (n=1 is excluded from Bernoulli)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let ode = expr!(diff(y, x) + y);

    let ode_type = ode.classify_ode(&y, &x);
    assert_ne!(
        ode_type,
        OdeType::Bernoulli,
        "y' + y = 0 should NOT be Bernoulli"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge cases: expressions that are NOT ODEs
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_no_derivative_returns_none() {
    // x + y = 0 has no derivative → not an ODE
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let expr = &x + &y;

    let result = expr.solve_ode(&y, &x);
    assert!(
        result.has_unevaluated(),
        "expression without derivative should return unevaluated DSolve"
    );
}

#[test]
fn comprehensive_pure_number_returns_none() {
    // 42 = 0 is not an ODE
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let expr = symplex::default_context().int(42);

    let result = expr.solve_ode(&y, &x);
    assert!(result.has_unevaluated(), "pure number should return unevaluated DSolve");
}

#[test]
fn comprehensive_no_derivative_classifies_unknown() {
    // x² + y² = 0 has no derivative
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let expr = &x.powi(2) + &y.powi(2);

    let ode_type = expr.classify_ode(&y, &x);
    assert_eq!(
        ode_type,
        OdeType::Unknown,
        "expression without derivative should classify as Unknown, got {ode_type:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// check_ode_solution: verify the symbolic solution checker
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_check_ode_solution_euler_cauchy() {
    // x²y'' − 2y = 0 → y = C1·x² + C2/x
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let two = symplex::default_context().int(2);
    let ode = &(&x_sq * &d2y) - &(&two * &y);

    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "should solve Euler-Cauchy");

    let verified = ode.check_ode_solution(&sol, &y, &x);
    if !verified {
        // Fall back to numerical verification — that's fine
        let c1 = symplex::default_context().symbol("C1");
        let c2 = symplex::default_context().symbol("C2");
        verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, POSITIVE_POINTS);
    }
}

#[test]
fn comprehensive_check_ode_solution_first_order_linear() {
    // y' + 2y = 0 → y = C1·exp(-2x)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let ode = expr!(diff(y, x) + 2 * y);

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y' + 2y = 0");

    let verified = ode.check_ode_solution(&sol, &y, &x);
    if !verified {
        let c1 = symplex::default_context().symbol("C1");
        verify_first_order_numerically(&ode, &sol, &[c1], &y, &x, FIRST_ORDER_POINTS);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Comprehensive regression: all basic types still solve
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_regression_all_basic_types() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);

    // SimpleSeparable: y' = x
    let ode1 = expr!(diff(y, x) - x);
    assert!(
        !ode1.solve_ode(&y, &x).has_unevaluated(),
        "y' = x should still work"
    );

    // FullSeparable: y' = xy
    let ode2 = expr!(diff(y, x) - x * y);
    assert!(
        !ode2.solve_ode(&y, &x).has_unevaluated(),
        "y' = xy should still work"
    );

    // FirstOrderLinearCC: y' + 5y = 0
    let ode3 = expr!(diff(y, x) + 5 * y);
    assert!(
        !ode3.solve_ode(&y, &x).has_unevaluated(),
        "y' + 5y = 0 should still work"
    );

    // FirstOrderLinearVC: y' + 2xy = 0
    let ode4 = expr!(diff(y, x) + 2 * x * y);
    assert!(
        !ode4.solve_ode(&y, &x).has_unevaluated(),
        "y' + 2xy = 0 should still work"
    );

    // SecondOrderLinearCCHomogeneous: y'' + y = 0
    let ode5 = &d2y + &y;
    assert!(
        !ode5.solve_ode(&y, &x).has_unevaluated(),
        "y'' + y = 0 should still work"
    );

    // SecondOrderLinearCCHomogeneous distinct: y'' - 3y' + 2y = 0
    let ode6 = &d2y - &(&dy * 3) + &(&y * 2);
    assert!(
        !ode6.solve_ode(&y, &x).has_unevaluated(),
        "y'' - 3y' + 2y = 0 should still work"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-point verification: confirm solutions at many points
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_multipoint_first_order_linear_cc() {
    // y' + 2y = 0 → y = C1·exp(-2x)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let ode = expr!(diff(y, x) + 2 * y);
    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y' + 2y = 0");

    // Verify at many points
    let c1 = symplex::default_context().symbol("C1");
    let many_points: Vec<(i64, i64)> = (1..=10).map(|i| (i, 4)).collect();
    verify_first_order_numerically(&ode, &sol, &[c1], &y, &x, &many_points);
}

#[test]
fn comprehensive_multipoint_second_order_distinct() {
    // y'' - 3y' + 2y = 0
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y - &(&dy * 3) + &(&y * 2);
    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve y'' - 3y' + 2y = 0");

    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    let many_points: Vec<(i64, i64)> = (1..=8).map(|i| (i, 10)).collect();
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, &many_points);
}

#[test]
fn comprehensive_multipoint_euler_cauchy() {
    // x²y'' − 2y = 0 → y = C1·x² + C2/x
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let x_sq = x.powi(2);
    let two = symplex::default_context().int(2);
    let ode = &(&x_sq * &d2y) - &(&two * &y);

    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("should solve Euler-Cauchy");

    // Verify at multiple positive x values
    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");
    let many_points: Vec<(i64, i64)> = (1..=6).map(|i| (i, 2)).collect();
    verify_second_order_numerically(&ode, &sol, &[c1, c2], &y, &x, &many_points);
}
