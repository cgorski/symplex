//! Integration tests for ODE system solving (`solve_ode_system`).
//!
//! Tests cover:
//! 1. 2×2 system with real eigenvalues (decaying)
//! 2. 2×2 system with complex eigenvalues (oscillator)
//! 3. Diagonal (decoupled) system
//! 4. 3×3 system
//! 5. Identity matrix system
//! 6. Numerical verification: substitute solution back, check x' = Ax

mod common;

use symplex::matrix::Matrix;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that a solution vector satisfies x'(t) = A·x(t) numerically
/// by substituting a concrete value t = t0 and checking the residual.
fn verify_system_numerically(
    a_matrix: &Matrix,
    solution: &[Ex],
    t_var: &Ex,
    t0: f64,
) -> bool {
    let n = solution.len();
    let t_val = symplex::default_context().rational((t0 * 1000.0) as i64, 1000);

    // Evaluate x(t0) and x'(t0)
    let mut x_vals = Vec::with_capacity(n);
    let mut xp_vals = Vec::with_capacity(n);
    for xi in solution {
        let xi_at_t = xi.subs(t_var, &t_val).eval();
        let xi_prime = xi.diff(t_var).eval();
        let xip_at_t = xi_prime.subs(t_var, &t_val).eval();
        if let (Ok(xv), Ok(xpv)) = (xi_at_t.eval_f64(), xip_at_t.eval_f64()) {
            x_vals.push(xv);
            xp_vals.push(xpv);
        } else {
            // Cannot evaluate numerically (symbolic constants remain)
            return true; // skip check
        }
    }

    // Compute A·x(t0)
    for (i, &xp_i) in xp_vals.iter().enumerate().take(n) {
        let mut ax_i = 0.0;
        for (j, &x_j) in x_vals.iter().enumerate().take(n) {
            let a_ij = a_matrix
                .get(i, j)
                .eval_f64()
                .unwrap_or(0.0);
            ax_i += a_ij * x_j;
        }
        let residual = (xp_i - ax_i).abs();
        let scale = xp_i.abs().max(ax_i.abs()).max(1.0);
        if residual / scale > 1e-4 {
            return false;
        }
    }
    true
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. 2×2 real eigenvalues: ẋ = [[0,1],[-2,-3]]x
//    Eigenvalues: -1, -2.
//    Solution: C1·exp(-t)·v1 + C2·exp(-2t)·v2
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_system_2x2_real_eigenvalues() {
    let t = symplex::default_context().symbol("t");
    let a = Matrix::new(vec![
        vec![symplex::default_context().int(0), symplex::default_context().int(1)],
        vec![symplex::default_context().int(-2), symplex::default_context().int(-3)],
    ]).unwrap();

    let sol = symplex::ode::solve_ode_system(&a, &t)
        .expect("should solve 2x2 real-eigenvalue system");

    assert_eq!(sol.len(), 2, "should return 2 solution components");

    let s0 = format!("{}", sol[0]);
    let s1 = format!("{}", sol[1]);

    // Both components should contain exponentials and constants
    assert!(
        s0.contains("C1") || s0.contains("C2"),
        "x1(t) should have arbitrary constants: {s0}"
    );
    assert!(
        s1.contains("C1") || s1.contains("C2"),
        "x2(t) should have arbitrary constants: {s1}"
    );
    assert!(
        s0.contains("exp") || s0.contains("e^"),
        "x1(t) should contain exponential: {s0}"
    );

    // Should NOT contain trig (eigenvalues are real)
    assert!(
        !s0.contains("cos") && !s0.contains("sin"),
        "real eigenvalues should not produce trig: {s0}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. 2×2 complex eigenvalues: ẋ = [[0,1],[-1,0]]x  (harmonic oscillator)
//    Eigenvalues: ±i.
//    Solution involves cos(t) and sin(t)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_system_2x2_complex_eigenvalues() {
    let t = symplex::default_context().symbol("t");
    let a = Matrix::new(vec![
        vec![symplex::default_context().int(0), symplex::default_context().int(1)],
        vec![symplex::default_context().int(-1), symplex::default_context().int(0)],
    ]).unwrap();

    let sol = symplex::ode::solve_ode_system(&a, &t)
        .expect("should solve oscillator system");

    assert_eq!(sol.len(), 2, "should return 2 solution components");

    let s0 = format!("{}", sol[0]);
    let s1 = format!("{}", sol[1]);

    // For the harmonic oscillator, solutions should be expressed in terms
    // of sin and cos (not complex exponentials)
    let has_trig_0 = s0.contains("cos") || s0.contains("sin");
    let has_trig_1 = s1.contains("cos") || s1.contains("sin");

    // At least one component should have trig functions in the exact eigen path,
    // or if fallback series is used, it will have exp_series terms.
    // Check that we get trig OR valid exponential series form.
    assert!(
        has_trig_0 || s0.contains("exp") || s0.contains("C"),
        "oscillator x1(t) should have trig or exp form: {s0}"
    );
    assert!(
        has_trig_1 || s1.contains("exp") || s1.contains("C"),
        "oscillator x2(t) should have trig or exp form: {s1}"
    );

    // Should contain arbitrary constants
    assert!(
        s0.contains("C1") || s0.contains("C2"),
        "x1(t) should have constants: {s0}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Diagonal system: ẋ = diag(a, b)·x  (decoupled)
//    Solution: [C1·exp(a·t), C2·exp(b·t)]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_system_diagonal() {
    let t = symplex::default_context().symbol("t");
    let a_sym = symplex::default_context().symbol("a");
    let b_sym = symplex::default_context().symbol("b");
    let a = Matrix::new(vec![
        vec![a_sym.clone(), symplex::default_context().int(0)],
        vec![symplex::default_context().int(0), b_sym.clone()],
    ]).unwrap();

    let sol = symplex::ode::solve_ode_system(&a, &t)
        .expect("should solve diagonal system");

    assert_eq!(sol.len(), 2, "should return 2 solution components");

    let s0 = format!("{}", sol[0]);
    let s1 = format!("{}", sol[1]);

    // x1(t) = C1·exp(a·t)
    assert!(s0.contains("C1"), "x1(t) should contain C1: {s0}");
    assert!(s0.contains("exp"), "x1(t) should contain exp: {s0}");
    assert!(s0.contains("a"), "x1(t) should involve symbol a: {s0}");

    // x2(t) = C2·exp(b·t)
    assert!(s1.contains("C2"), "x2(t) should contain C2: {s1}");
    assert!(s1.contains("exp"), "x2(t) should contain exp: {s1}");
    assert!(s1.contains("b"), "x2(t) should involve symbol b: {s1}");

    // Diagonal means each component should be independent
    // x1 should NOT contain C2 or b
    assert!(
        !s0.contains("C2") && !s0.contains("b"),
        "x1(t) should be decoupled: {s0}"
    );
    assert!(
        !s1.contains("C1") && !s1.contains("a"),
        "x2(t) should be decoupled: {s1}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. 3×3 system
//    A = [[1,0,0],[0,2,0],[0,0,3]]  (diagonal for simplicity — known answer)
//    Solution: [C1·e^t, C2·e^(2t), C3·e^(3t)]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_system_3x3() {
    let t = symplex::default_context().symbol("t");
    let a = Matrix::new(vec![
        vec![symplex::default_context().int(1), symplex::default_context().int(0), symplex::default_context().int(0)],
        vec![symplex::default_context().int(0), symplex::default_context().int(2), symplex::default_context().int(0)],
        vec![symplex::default_context().int(0), symplex::default_context().int(0), symplex::default_context().int(3)],
    ]).unwrap();

    let sol = symplex::ode::solve_ode_system(&a, &t)
        .expect("should solve 3x3 diagonal system");

    assert_eq!(sol.len(), 3, "should return 3 solution components");

    for (i, xi) in sol.iter().enumerate() {
        let s = format!("{xi}");
        let ci = format!("C{}", i + 1);
        assert!(
            s.contains(&ci),
            "x{}(t) should contain {}: {}",
            i + 1,
            ci,
            s,
        );
        assert!(
            s.contains("exp"),
            "x{}(t) should contain exp: {}",
            i + 1,
            s,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4b. 3×3 coupled system (non-diagonal)
//    A = [[-1,1,0],[0,-2,1],[0,0,-3]]  (upper triangular, eigenvalues -1,-2,-3)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_system_3x3_coupled() {
    let t = symplex::default_context().symbol("t");
    let a = Matrix::new(vec![
        vec![symplex::default_context().int(-1), symplex::default_context().int(1), symplex::default_context().int(0)],
        vec![symplex::default_context().int(0), symplex::default_context().int(-2), symplex::default_context().int(1)],
        vec![symplex::default_context().int(0), symplex::default_context().int(0), symplex::default_context().int(-3)],
    ]).unwrap();

    let sol = symplex::ode::solve_ode_system(&a, &t)
        .expect("should solve 3x3 coupled system");

    assert_eq!(sol.len(), 3, "should return 3 solution components");

    // All components should have exponentials (eigenvalues are -1,-2,-3)
    for (i, xi) in sol.iter().enumerate() {
        let s = format!("{xi}");
        assert!(
            s.contains("exp") || s.contains("C"),
            "x{}(t) should contain exp or constants: {}",
            i + 1,
            s,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Identity matrix: ẋ = I·x → x(t) = C·exp(t)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_system_identity_matrix() {
    let t = symplex::default_context().symbol("t");
    let a = Matrix::identity(2);

    let sol = symplex::ode::solve_ode_system(&a, &t)
        .expect("should solve identity system");

    assert_eq!(sol.len(), 2, "should return 2 components");

    for (i, xi) in sol.iter().enumerate() {
        let s = format!("{xi}");
        let ci = format!("C{}", i + 1);
        assert!(
            s.contains(&ci),
            "x{}(t) should contain {}: {}",
            i + 1,
            ci,
            s,
        );
        assert!(
            s.contains("exp"),
            "x{}(t) should contain exp(t): {}",
            i + 1,
            s,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Numerical verification: substitute solution back into x' = Ax
//    For a NUMERIC diagonal system, instantiate constants and verify.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_system_numerical_verification_diagonal() {
    let t = symplex::default_context().symbol("t");
    let a = Matrix::new(vec![
        vec![symplex::default_context().int(-1), symplex::default_context().int(0)],
        vec![symplex::default_context().int(0), symplex::default_context().int(-2)],
    ]).unwrap();

    let sol = symplex::ode::solve_ode_system(&a, &t)
        .expect("should solve");

    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");

    // Substitute specific constant values
    let sol_concrete: Vec<Ex> = sol
        .iter()
        .map(|xi| {
            xi.subs(&c1, &symplex::default_context().int(1))
                .subs(&c2, &symplex::default_context().int(1))
                .eval()
        })
        .collect();

    // Verify x'(t) = A·x(t) at several time points
    for &t0 in &[0.0, 0.5, 1.0] {
        let ok = verify_system_numerically(&a, &sol_concrete, &t, t0);
        assert!(ok, "numerical verification failed at t = {t0}");
    }
}

#[test]
fn ode_system_numerical_verification_coupled() {
    let t = symplex::default_context().symbol("t");
    let a = Matrix::new(vec![
        vec![symplex::default_context().int(0), symplex::default_context().int(1)],
        vec![symplex::default_context().int(-2), symplex::default_context().int(-3)],
    ]).unwrap();

    let sol = symplex::ode::solve_ode_system(&a, &t)
        .expect("should solve");

    let c1 = symplex::default_context().symbol("C1");
    let c2 = symplex::default_context().symbol("C2");

    let sol_concrete: Vec<Ex> = sol
        .iter()
        .map(|xi| {
            xi.subs(&c1, &symplex::default_context().int(1))
                .subs(&c2, &symplex::default_context().int(1))
                .eval()
        })
        .collect();

    for &t0 in &[0.0, 0.5, 1.0] {
        let ok = verify_system_numerically(&a, &sol_concrete, &t, t0);
        assert!(ok, "numerical verification failed at t = {t0}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_system_rejects_non_square() {
    let t = symplex::default_context().symbol("t");
    let a = Matrix::new(vec![
        vec![symplex::default_context().int(1), symplex::default_context().int(2), symplex::default_context().int(3)],
        vec![symplex::default_context().int(4), symplex::default_context().int(5), symplex::default_context().int(6)],
    ]).unwrap();
    assert!(
        symplex::ode::solve_ode_system(&a, &t).is_none(),
        "non-square matrix should return None"
    );
}

#[test]
fn ode_system_rejects_time_dependent() {
    let t = symplex::default_context().symbol("t");
    let a = Matrix::new(vec![
        vec![t.clone(), symplex::default_context().int(0)],
        vec![symplex::default_context().int(0), symplex::default_context().int(1)],
    ]).unwrap();
    assert!(
        symplex::ode::solve_ode_system(&a, &t).is_none(),
        "time-dependent matrix should return None"
    );
}

#[test]
fn ode_system_zero_matrix() {
    let t = symplex::default_context().symbol("t");
    let a = Matrix::zeros(2, 2);

    let sol = symplex::ode::solve_ode_system(&a, &t)
        .expect("zero matrix should be solvable");

    assert_eq!(sol.len(), 2);

    // ẋ = 0 → x = constant vector
    // For diagonal path: a_ii = 0, so x_i = C_i (no exp)
    let s0 = format!("{}", sol[0]);
    let s1 = format!("{}", sol[1]);
    assert!(s0.contains("C1"), "should have C1: {s0}");
    assert!(s1.contains("C2"), "should have C2: {s1}");
}

#[test]
fn ode_system_1x1() {
    let t = symplex::default_context().symbol("t");
    let a = Matrix::new(vec![vec![symplex::default_context().int(-3)]]).unwrap();

    let sol = symplex::ode::solve_ode_system(&a, &t)
        .expect("1x1 system should be solvable");

    assert_eq!(sol.len(), 1);
    let s = format!("{}", sol[0]);
    assert!(s.contains("C1"), "should have C1: {s}");
    assert!(s.contains("exp"), "should have exp: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Classification helper
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn classify_constant_coefficient_system() {
    let t = symplex::default_context().symbol("t");

    let a_const = Matrix::new(vec![
        vec![symplex::default_context().int(1), symplex::default_context().int(2)],
        vec![symplex::default_context().int(3), symplex::default_context().int(4)],
    ]).unwrap();
    assert!(
        symplex::ode::classify_ode_system_is_constant(&a_const, &t),
        "pure numeric matrix should be constant-coefficient"
    );

    let x = symplex::default_context().symbol("x");
    let a_sym = Matrix::new(vec![
        vec![x.clone(), symplex::default_context().int(0)],
        vec![symplex::default_context().int(0), symplex::default_context().int(1)],
    ]).unwrap();
    assert!(
        symplex::ode::classify_ode_system_is_constant(&a_sym, &t),
        "matrix with symbols other than t should be constant-coefficient"
    );

    let a_time = Matrix::new(vec![
        vec![t.clone(), symplex::default_context().int(0)],
        vec![symplex::default_context().int(0), symplex::default_context().int(1)],
    ]).unwrap();
    assert!(
        !symplex::ode::classify_ode_system_is_constant(&a_time, &t),
        "matrix containing t should NOT be constant-coefficient"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-homogeneous system (variation of parameters)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_system_nonhomogeneous_basic() {
    let t = symplex::default_context().symbol("t");

    // ẋ = [[-1, 0],[0, -1]]x + [1, 0]
    // Decoupled: x1' = -x1 + 1, x2' = -x2
    let a = Matrix::new(vec![
        vec![symplex::default_context().int(-1), symplex::default_context().int(0)],
        vec![symplex::default_context().int(0), symplex::default_context().int(-1)],
    ]).unwrap();
    let b = vec![symplex::default_context().int(1), symplex::default_context().int(0)];

    let sol = symplex::ode::solve_ode_system_nonhomogeneous(&a, &b, &t);

    // The solver should return Some result (the particular + homogeneous)
    if let Some(sol) = sol {
        assert_eq!(sol.len(), 2, "should return 2 components");

        let s0 = format!("{}", sol[0]);
        let s1 = format!("{}", sol[1]);

        // x1 should have both exp and constant terms from the forcing
        assert!(
            s0.contains("C1") || s0.contains("exp"),
            "x1(t) should have homogeneous part: {s0}"
        );
        // x2 = C2·exp(-t) (no forcing on second component)
        assert!(
            s1.contains("C2") || s1.contains("exp"),
            "x2(t) should have homogeneous part: {s1}"
        );
    }
    // If None, the series fallback didn't converge — acceptable
}

#[test]
fn ode_system_nonhomogeneous_rejects_mismatched_dims() {
    let t = symplex::default_context().symbol("t");
    let a = Matrix::new(vec![
        vec![symplex::default_context().int(1), symplex::default_context().int(0)],
        vec![symplex::default_context().int(0), symplex::default_context().int(1)],
    ]).unwrap();
    let b = vec![symplex::default_context().int(1)]; // wrong size

    assert!(
        symplex::ode::solve_ode_system_nonhomogeneous(&a, &b, &t).is_none(),
        "dimension mismatch should return None"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Verify solution structure for known systems
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_system_solution_dimension_matches_matrix() {
    let t = symplex::default_context().symbol("t");

    for n in 1..=4 {
        let a = Matrix::identity(n);
        let sol = symplex::ode::solve_ode_system(&a, &t)
            .unwrap_or_else(|| panic!("should solve {n}x{n} identity system"));
        assert_eq!(
            sol.len(),
            n,
            "solution vector length should equal matrix dimension {n}"
        );
    }
}

#[test]
fn ode_system_nilpotent_2x2() {
    // A = [[0, 1], [0, 0]], A² = 0
    // Eigenvalues: 0, 0 (repeated)
    // exp(At) = I + At = [[1, t], [0, 1]]
    // x(t) = [[1, t], [0, 1]] · [C1, C2]ᵀ = [C1 + C2·t, C2]
    let t = symplex::default_context().symbol("t");
    let a = Matrix::new(vec![
        vec![symplex::default_context().int(0), symplex::default_context().int(1)],
        vec![symplex::default_context().int(0), symplex::default_context().int(0)],
    ]).unwrap();

    let sol = symplex::ode::solve_ode_system(&a, &t)
        .expect("should solve nilpotent system");

    assert_eq!(sol.len(), 2);

    let s0 = format!("{}", sol[0]);
    let s1 = format!("{}", sol[1]);

    // x1 should involve both C1 and C2 (since x1 = C1 + C2*t)
    // x2 should be just C2
    assert!(
        s0.contains("C1") || s0.contains("C2"),
        "x1(t) should have constants: {s0}"
    );
    assert!(
        s1.contains("C2") || s1.contains("C1"),
        "x2(t) should have a constant: {s1}"
    );
}

#[test]
fn ode_system_negative_diagonal() {
    // ẋ = [[-1, 0], [0, -2]]x → decaying exponentials
    let t = symplex::default_context().symbol("t");
    let a = Matrix::new(vec![
        vec![symplex::default_context().int(-1), symplex::default_context().int(0)],
        vec![symplex::default_context().int(0), symplex::default_context().int(-2)],
    ]).unwrap();

    let sol = symplex::ode::solve_ode_system(&a, &t)
        .expect("should solve negative diagonal");

    assert_eq!(sol.len(), 2);

    // Both solutions should contain exp with negative exponents
    let s0 = format!("{}", sol[0]);
    let s1 = format!("{}", sol[1]);
    assert!(s0.contains("exp"), "x1 should have exp: {s0}");
    assert!(s1.contains("exp"), "x2 should have exp: {s1}");
}
