//! Integration tests for polynomial system solving with irrational roots.
//!
//! These tests verify that `solve_system` (backed by Gröbner basis +
//! back-substitution) correctly finds irrational roots by falling back
//! to the general symbolic solver when the rational root theorem returns
//! empty.

use num_bigint::BigInt;
use num_rational::Ratio;
use symplex::multipoly::{GrevLex, MultiPoly};
use symplex::polysys::solve_polynomial_system;

// ─── helpers ───────────────────────────────────────────────────────────────

fn rat(n: i64) -> Ratio<BigInt> {
    Ratio::from_integer(BigInt::from(n))
}

/// Evaluate an expression to f64, returning None on failure.
fn eval(e: &symplex::expr::Ex) -> Option<f64> {
    e.eval_f64().ok()
}

/// Check that a value is approximately zero.
fn approx_zero(val: f64, tol: f64) -> bool {
    val.abs() < tol
}

// ═══════════════════════════════════════════════════════════════════════════
// x² + y² = 3, x = y → should find ±√(3/2) solutions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_system_irrational_circle_diagonal() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");

    let eq1 = &x.powi(2) + &y.powi(2) - 3;
    let eq2 = &x - &y;

    let solutions = symplex::polysys::solve_system_ex(&[eq1, eq2], &[x.clone(), y.clone()]);

    match solutions {
        Ok(sols) => {
            assert!(
                !sols.is_empty(),
                "x²+y²=3, x=y should have solutions (irrational), got empty"
            );

            // Each solution should satisfy both equations numerically.
            for sol in &sols {
                assert_eq!(sol.len(), 2, "each solution should have 2 values");

                let x_val = eval(&sol[0]);
                let y_val = eval(&sol[1]);

                if let (Some(xv), Some(yv)) = (x_val, y_val) {
                    // Check x = y
                    assert!(
                        approx_zero(xv - yv, 1e-10),
                        "x should equal y: x={xv}, y={yv}"
                    );
                    // Check x² + y² = 3
                    let lhs = xv * xv + yv * yv;
                    assert!(
                        approx_zero(lhs - 3.0, 1e-10),
                        "x²+y² should be 3, got {lhs}"
                    );
                }
            }

            // Should have exactly 2 solutions (positive and negative √(3/2)).
            assert_eq!(
                sols.len(),
                2,
                "expected 2 solutions for x²+y²=3, x=y, got {}",
                sols.len()
            );
        }
        Err(e) => {
            panic!("solve_system should succeed for x²+y²=3, x=y: {e}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// x² + y² = 5, x*y = 2 → rational solutions (1,2), (2,1), (-1,-2), (-2,-1)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_system_rational_two_conics() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");

    let eq1 = &x.powi(2) + &y.powi(2) - 5;
    let eq2 = &x * &y - 2;

    let solutions = symplex::polysys::solve_system_ex(&[eq1, eq2], &[x.clone(), y.clone()]);

    match solutions {
        Ok(sols) => {
            assert_eq!(
                sols.len(),
                4,
                "expected 4 solutions for x²+y²=5, xy=2, got {}",
                sols.len()
            );

            // Verify each solution numerically.
            for sol in &sols {
                let x_val = eval(&sol[0]);
                let y_val = eval(&sol[1]);

                if let (Some(xv), Some(yv)) = (x_val, y_val) {
                    let eq1_val = xv * xv + yv * yv - 5.0;
                    let eq2_val = xv * yv - 2.0;
                    assert!(
                        approx_zero(eq1_val, 1e-10),
                        "x²+y²-5 should ≈ 0, got {eq1_val} for x={xv}, y={yv}"
                    );
                    assert!(
                        approx_zero(eq2_val, 1e-10),
                        "xy-2 should ≈ 0, got {eq2_val} for x={xv}, y={yv}"
                    );
                }
            }
        }
        Err(e) => {
            panic!("solve_system should succeed for x²+y²=5, xy=2: {e}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// x² = 2 (univariate via Gröbner) → √2, -√2
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_system_univariate_irrational() {
    let x = symplex::default_context().symbol("x");

    let eq = &x.powi(2) - 2;

    let solutions = symplex::polysys::solve_system_ex(&[eq], std::slice::from_ref(&x));

    match solutions {
        Ok(sols) => {
            assert!(
                !sols.is_empty(),
                "x²=2 should have solutions (±√2), got empty"
            );

            for sol in &sols {
                let x_val = eval(&sol[0]);
                if let Some(xv) = x_val {
                    let check = xv * xv - 2.0;
                    assert!(
                        approx_zero(check, 1e-10),
                        "x² should be 2, got x={xv}, x²={}", xv * xv
                    );
                }
            }

            assert_eq!(
                sols.len(),
                2,
                "x²=2 should have 2 solutions, got {}",
                sols.len()
            );
        }
        Err(e) => {
            panic!("solve_system should succeed for x²=2: {e}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Low-level: rational root theorem returns empty for irrational poly
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn polysys_rational_root_empty_for_x2_minus_2() {
    // The rational-only solver should return empty for x²-2=0.
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let p = &x * &x - MultiPoly::from_int(1, 2);

    let sols = solve_polynomial_system(&[p]).unwrap();
    // This returns empty because ±√2 are irrational.
    assert!(
        sols.is_empty(),
        "rational-only solver should return empty for x²-2, got {:?}",
        sols
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Low-level: rational root theorem works for rational roots
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn polysys_rational_root_finds_rationals() {
    // x² - 4 = 0 → x = ±2 (rational roots exist).
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let p = &x * &x - MultiPoly::from_int(1, 4);

    let sols = solve_polynomial_system(&[p]).unwrap();
    assert_eq!(sols.len(), 2, "x²-4 should have 2 rational roots");

    let mut root_vals: Vec<i64> = sols
        .iter()
        .map(|s| s[0].to_integer().try_into().unwrap())
        .collect();
    root_vals.sort();
    assert_eq!(root_vals, vec![-2, 2]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Verify no regression: circle-line system (rational solutions)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn polysys_circle_line_no_regression() {
    // x² + y² = 1, x + y = 1 → (0,1), (1,0)
    let nv = 2;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);
    let one = MultiPoly::from_int(nv, 1);

    let circle = &(&x * &x) + &(&y * &y) - one.clone();
    let line = &x + &y - one;

    let sols = solve_polynomial_system(&[circle, line]).unwrap();
    assert_eq!(
        sols.len(),
        2,
        "circle-line should have 2 rational solutions"
    );

    // Verify each solution.
    for sol in &sols {
        let xv = &sol[0];
        let yv = &sol[1];
        // x² + y² = 1
        let check = xv * xv + yv * yv;
        assert!(
            check == rat(1),
            "x²+y²=1 check failed: got {check}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Verify symbolic fallback for 2-var system with irrational roots
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_system_ex_irrational_symmetric() {
    // x² + y² = 2, x = y → x = y = ±1 (these are actually rational!)
    // But this verifies the fallback path doesn't break rational-solution systems.
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");

    let eq1 = &x.powi(2) + &y.powi(2) - 2;
    let eq2 = &x - &y;

    let solutions = symplex::polysys::solve_system_ex(&[eq1, eq2], &[x.clone(), y.clone()]);

    match solutions {
        Ok(sols) => {
            assert!(
                !sols.is_empty(),
                "x²+y²=2, x=y should have solutions, got empty"
            );

            for sol in &sols {
                if let (Some(xv), Some(yv)) = (eval(&sol[0]), eval(&sol[1])) {
                    assert!(
                        approx_zero(xv - yv, 1e-10),
                        "x should equal y"
                    );
                    assert!(
                        approx_zero(xv * xv + yv * yv - 2.0, 1e-10),
                        "x²+y² should be 2"
                    );
                }
            }
        }
        Err(e) => {
            panic!("solve_system should succeed: {e}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge case: system with no real solutions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_system_no_real_solutions() {
    // x² + 1 = 0 has no real solutions.
    let x = symplex::default_context().symbol("x");
    let eq = &x.powi(2) + 1;

    let solutions = symplex::polysys::solve_system_ex(&[eq], std::slice::from_ref(&x));

    match solutions {
        Ok(sols) => {
            // The solver may return complex solutions or empty.
            // With the symbolic fallback, it might find ±i.
            // Either way, any returned solution should satisfy the equation.
            for sol in &sols {
                let x_val = eval(&sol[0]);
                // Complex solutions won't evaluate to real f64,
                // so eval might fail — that's acceptable.
                if let Some(xv) = x_val {
                    let check = xv * xv + 1.0;
                    assert!(
                        approx_zero(check, 1e-10),
                        "if a real root is returned, it must satisfy x²+1=0"
                    );
                }
            }
        }
        Err(_) => {
            // Also acceptable: the system might be flagged as having
            // no real solutions.
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear system: verify no regression
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_system_linear_no_regression() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");

    // x + y = 1, x - y = 0 → x = y = 1/2
    let eq1 = &x + &y - 1;
    let eq2 = &x - &y;

    let solutions = symplex::polysys::solve_system_ex(&[eq1, eq2], &[x.clone(), y.clone()]).unwrap();

    assert_eq!(solutions.len(), 1, "linear system should have 1 solution");

    let x_str = format!("{}", solutions[0][0]);
    let y_str = format!("{}", solutions[0][1]);

    assert_eq!(x_str, "1/2", "x should be 1/2, got {x_str}");
    assert_eq!(y_str, "1/2", "y should be 1/2, got {y_str}");
}
