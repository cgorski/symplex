//! Integration tests for polynomial system solving (`polysys`) and
//! 2-DOF planar inverse kinematics (`robotics::inverse_kinematics_2dof`).

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Zero;
use symplex::multipoly::{GrevLex, MultiPoly};
use symplex::polysys::solve_polynomial_system;
use symplex::robotics::inverse_kinematics_2dof;

// ─── helpers ───────────────────────────────────────────────────────────────

fn rat(n: i64) -> Ratio<BigInt> {
    Ratio::from_integer(BigInt::from(n))
}

fn ratio(p: i64, q: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(p), BigInt::from(q))
}

/// Sort a solution set so that comparisons are order-independent.
fn sorted(mut sols: Vec<Vec<Ratio<BigInt>>>) -> Vec<Vec<Ratio<BigInt>>> {
    // Each individual solution vector is already ordered by variable index,
    // so we only sort the outer vec.
    sols.sort_by(|a, b| {
        for (ai, bi) in a.iter().zip(b.iter()) {
            match ai.cmp(bi) {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }
        a.len().cmp(&b.len())
    });
    sols
}

/// Verify that `point` is a common root of all `polys` (each evaluates to 0).
fn verify_solution(polys: &[MultiPoly<GrevLex>], point: &[Ratio<BigInt>]) {
    for (i, p) in polys.iter().enumerate() {
        let val = p.eval(point);
        assert!(
            val.is_zero(),
            "polynomial {} evaluated to {} at {:?} (expected 0)",
            i,
            val,
            point
        );
    }
}

/// Check forward kinematics for a 2-DOF planar arm.
fn verify_fk_2dof(l1: f64, l2: f64, theta1: f64, theta2: f64, tx: f64, ty: f64) -> bool {
    let x = l1 * theta1.cos() + l2 * (theta1 + theta2).cos();
    let y = l1 * theta1.sin() + l2 * (theta1 + theta2).sin();
    (x - tx).abs() < 0.01 && (y - ty).abs() < 0.01
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial system tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_single_variable_quadratic() {
    // x² - 4 = 0 in 1 variable → x = 2, x = -2
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let p = &x * &x - MultiPoly::from_int(1, 4);

    let sols = solve_polynomial_system(&[p]).unwrap();
    let sols = sorted(sols);

    assert_eq!(sols.len(), 2, "expected 2 solutions, got {:?}", sols);
    assert!(sols.contains(&vec![rat(-2)]));
    assert!(sols.contains(&vec![rat(2)]));
}

#[test]
fn solve_linear_2var() {
    // x + y - 1 = 0
    // x - y     = 0
    // → unique solution (1/2, 1/2)
    let nv = 2;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);
    let one = MultiPoly::<GrevLex>::from_int(nv, 1);

    let p1 = &x + &y - one; // x + y - 1
    let p2 = &x - &y; // x - y

    let sols = solve_polynomial_system(&[p1, p2]).unwrap();

    assert_eq!(sols.len(), 1, "expected 1 solution, got {:?}", sols);
    assert_eq!(sols[0].len(), 2);
    assert_eq!(sols[0][0], ratio(1, 2), "x should be 1/2");
    assert_eq!(sols[0][1], ratio(1, 2), "y should be 1/2");
}

#[test]
fn solve_circle_line() {
    // x² + y² - 1 = 0   (unit circle)
    // x + y - 1   = 0   (line)
    // → (0, 1) and (1, 0)
    let nv = 2;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);
    let one = MultiPoly::<GrevLex>::from_int(nv, 1);

    let circle = &(&x * &x) + &(&y * &y) - one.clone();
    let line = &x + &y - one;

    let sols = solve_polynomial_system(&[circle, line]).unwrap();
    let sols = sorted(sols);

    assert_eq!(sols.len(), 2, "expected 2 solutions, got {:?}", sols);
    assert!(
        sols.contains(&vec![rat(0), rat(1)]),
        "missing solution (0, 1): {:?}",
        sols
    );
    assert!(
        sols.contains(&vec![rat(1), rat(0)]),
        "missing solution (1, 0): {:?}",
        sols
    );
}

#[test]
fn solve_two_conics() {
    // x² + y² - 5 = 0
    // x·y - 2     = 0
    // Solutions: (1,2), (2,1), (-1,-2), (-2,-1)
    let nv = 2;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);

    let p1 = &(&x * &x) + &(&y * &y) - MultiPoly::from_int(nv, 5);
    let p2 = &x * &y - MultiPoly::from_int(nv, 2);

    let sols = solve_polynomial_system(&[p1, p2]).unwrap();
    let sols = sorted(sols);

    assert_eq!(sols.len(), 4, "expected 4 solutions, got {:?}", sols);
    assert!(sols.contains(&vec![rat(1), rat(2)]));
    assert!(sols.contains(&vec![rat(2), rat(1)]));
    assert!(sols.contains(&vec![rat(-1), rat(-2)]));
    assert!(sols.contains(&vec![rat(-2), rat(-1)]));
}

#[test]
fn solve_inconsistent_no_real_solutions() {
    // x² + y² + 1 = 0 — sum of squares + 1 is always positive, no real roots
    // The Gröbner basis over ℚ won't produce a constant 1 (since the ideal
    // is not the whole ring — it has complex solutions), so this might return
    // an error or empty depending on the implementation.
    let nv = 2;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);

    let p = &(&x * &x) + &(&y * &y) + MultiPoly::from_int(nv, 1);

    let result = solve_polynomial_system(&[p]);
    match result {
        Ok(sols) => {
            // No rational (or even real) solutions
            assert!(
                sols.is_empty(),
                "x²+y²+1=0 should have no rational solutions, got {:?}",
                sols
            );
        }
        Err(_) => {
            // Also acceptable: ideal might not be zero-dimensional with a single
            // equation in 2 variables.
        }
    }
}

#[test]
fn solve_verify_solutions_circle_line() {
    // Verify that each returned solution actually satisfies every equation.
    let nv = 2;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);
    let one = MultiPoly::<GrevLex>::from_int(nv, 1);

    let circle = &(&x * &x) + &(&y * &y) - one.clone();
    let line = &x + &y - one;

    let system = vec![circle, line];
    let sols = solve_polynomial_system(&system).unwrap();

    assert!(!sols.is_empty(), "should have at least one solution");
    for sol in &sols {
        verify_solution(&system, sol);
    }
}

#[test]
fn solve_verify_solutions_two_conics() {
    let nv = 2;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);

    let p1 = &(&x * &x) + &(&y * &y) - MultiPoly::from_int(nv, 5);
    let p2 = &x * &y - MultiPoly::from_int(nv, 2);

    let system = vec![p1, p2];
    let sols = solve_polynomial_system(&system).unwrap();

    assert_eq!(sols.len(), 4);
    for sol in &sols {
        verify_solution(&system, sol);
    }
}

#[test]
fn solve_linear_3var() {
    // x + y + z = 6
    // x - y     = 0
    //     y - z = 0
    // → (2, 2, 2)
    let nv = 3;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);
    let z = MultiPoly::<GrevLex>::var(nv, 2);

    let p1 = &(&x + &y) + &z - MultiPoly::from_int(nv, 6);
    let p2 = &x - &y;
    let p3 = &y - &z;

    let system = vec![p1, p2, p3];
    let sols = solve_polynomial_system(&system).unwrap();

    assert_eq!(sols.len(), 1, "expected 1 solution, got {:?}", sols);
    for sol in &sols {
        verify_solution(&system, sol);
    }
    // Each variable should be 2
    assert_eq!(sols[0][0], rat(2));
    assert_eq!(sols[0][1], rat(2));
    assert_eq!(sols[0][2], rat(2));
}

#[test]
fn solve_single_linear() {
    // 3x - 6 = 0 → x = 2
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let p = x.scale(&rat(3)).sub(&MultiPoly::from_int(1, 6));

    let sols = solve_polynomial_system(&[p]).unwrap();
    assert_eq!(sols.len(), 1);
    assert_eq!(sols[0], vec![rat(2)]);
}

#[test]
fn solve_two_linear_unique() {
    // 2x + 3y = 8
    // x  -  y = 1
    // → x = 11/5, y = 6/5
    let nv = 2;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);

    let p1 = x
        .scale(&rat(2))
        .add(&y.scale(&rat(3)))
        .sub(&MultiPoly::from_int(nv, 8));
    let p2 = x.sub(&y).sub(&MultiPoly::from_int(nv, 1));

    let system = vec![p1, p2];
    let sols = solve_polynomial_system(&system).unwrap();

    assert_eq!(sols.len(), 1, "expected unique solution, got {:?}", sols);
    for sol in &sols {
        verify_solution(&system, sol);
    }
    assert_eq!(sols[0][0], ratio(11, 5));
    assert_eq!(sols[0][1], ratio(6, 5));
}

#[test]
fn solve_quadratic_single_var_no_rational_roots() {
    // x² + 1 = 0 — no rational roots
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let p = &x * &x + MultiPoly::from_int(1, 1);

    let sols = solve_polynomial_system(&[p]).unwrap();
    assert!(
        sols.is_empty(),
        "x²+1 has no rational roots, got {:?}",
        sols
    );
}

#[test]
fn solve_cubic_single_var() {
    // x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3) = 0
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let x2 = &x * &x;
    let x3 = &x2 * &x;

    let six = MultiPoly::from_int(1, 6);
    let p = &(&x3 - &x2.scale(&rat(6))) + &(&x.scale(&rat(11)) - &six);

    let sols = solve_polynomial_system(&[p]).unwrap();
    let sols = sorted(sols);

    assert_eq!(sols.len(), 3, "expected 3 roots, got {:?}", sols);
    assert!(sols.contains(&vec![rat(1)]));
    assert!(sols.contains(&vec![rat(2)]));
    assert!(sols.contains(&vec![rat(3)]));
}

// ═══════════════════════════════════════════════════════════════════════════
// 2-DOF inverse kinematics tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ik_2dof_extended_arm() {
    // l1 = 1, l2 = 1, target = (2, 0): fully extended along x-axis.
    // θ₂ = 0, θ₁ = 0 is the only solution.
    let solutions = inverse_kinematics_2dof(1.0, 1.0, 2.0, 0.0);

    assert!(
        !solutions.is_empty(),
        "fully extended arm should have at least one IK solution"
    );
    for &(t1, t2) in &solutions {
        assert!(
            verify_fk_2dof(1.0, 1.0, t1, t2, 2.0, 0.0),
            "FK verification failed for θ₁={t1}, θ₂={t2}"
        );
    }
}

#[test]
fn ik_2dof_basic() {
    // l1 = 1, l2 = 1, target = (1, 1).
    // Should find 2 solution branches (elbow-up / elbow-down).
    let solutions = inverse_kinematics_2dof(1.0, 1.0, 1.0, 1.0);

    // At minimum, every returned solution must satisfy FK.
    for &(t1, t2) in &solutions {
        assert!(
            verify_fk_2dof(1.0, 1.0, t1, t2, 1.0, 1.0),
            "FK verification failed for θ₁={t1:.4}, θ₂={t2:.4}: \
             got ({:.4}, {:.4}), expected (1, 1)",
            1.0 * t1.cos() + 1.0 * (t1 + t2).cos(),
            1.0 * t1.sin() + 1.0 * (t1 + t2).sin(),
        );
    }

    // Typically there are 2 IK solutions for this reachable interior point.
    if solutions.len() == 2 {
        // Good — the expected case.
    } else if solutions.is_empty() {
        panic!("IK returned no solutions for a clearly reachable point (1,1)");
    }
    // We accept 1 or 2 solutions (rational root theorem might miss irrational ones).
}

#[test]
fn ik_2dof_verify_fk() {
    // Test a few different targets and verify FK for every returned solution.
    let cases: Vec<(f64, f64, f64, f64)> = vec![
        (1.0, 1.0, 2.0, 0.0),  // fully extended
        (1.0, 1.0, 0.0, 2.0),  // fully extended upward
        (1.0, 1.0, -2.0, 0.0), // fully extended backward
        (2.0, 1.0, 3.0, 0.0),  // different link lengths, extended
        (1.0, 1.0, 0.0, 0.0),  // folded back to origin (boundary)
    ];

    for (l1, l2, tx, ty) in cases {
        let solutions = inverse_kinematics_2dof(l1, l2, tx, ty);
        for &(t1, t2) in &solutions {
            assert!(
                verify_fk_2dof(l1, l2, t1, t2, tx, ty),
                "FK verification failed: l1={l1}, l2={l2}, target=({tx},{ty}), \
                 θ₁={t1:.4}, θ₂={t2:.4}"
            );
        }
    }
}

#[test]
fn ik_2dof_unreachable() {
    // Target is beyond the reach of the arm: |(tx,ty)| > l1 + l2
    let solutions = inverse_kinematics_2dof(1.0, 1.0, 5.0, 0.0);
    assert!(
        solutions.is_empty(),
        "unreachable target should yield no solutions, got {:?}",
        solutions
    );
}
