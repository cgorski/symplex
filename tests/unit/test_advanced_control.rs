//! Integration tests for advanced matrix decompositions and control algorithms:
//! Cholesky decomposition, pseudo-inverse, Riccati residual, Ackermann pole placement.

use super::common;
use symplex::prelude::*;

use symplex::control::StateSpace;
use symplex::matrix::Matrix;

// ═══════════════════════════════════════════════════════════════════════════
// Helper
// ═══════════════════════════════════════════════════════════════════════════

/// Check that every element of a matrix evaluates to approximately `expected`
/// at the given tolerance. `expected` is a flat row-major array.
fn assert_matrix_approx(mat: &Matrix, expected: &[f64], tol: f64) {
    let (nr, nc) = mat.shape();
    assert_eq!(
        nr * nc,
        expected.len(),
        "expected {} elements, matrix has {}",
        expected.len(),
        nr * nc
    );
    for i in 0..nr {
        for j in 0..nc {
            let val = mat.get(i, j).simplify().eval_f64().unwrap_or_else(|e| {
                panic!("evalf_f64 failed at ({i},{j}): {e}");
            });
            let exp = expected[i * nc + j];
            assert!(
                common::approx_eq(val, exp, tol),
                "Mismatch at ({i},{j}): got {val}, expected {exp} (tol={tol})"
            );
        }
    }
}

/// Check that every element of a matrix evaluates to approximately zero.
fn assert_matrix_near_zero(mat: &Matrix, tol: f64) {
    let (nr, nc) = mat.shape();
    for i in 0..nr {
        for j in 0..nc {
            let val = mat.get(i, j).simplify().eval_f64().unwrap_or_else(|e| {
                panic!("evalf_f64 failed at ({i},{j}): {e}");
            });
            assert!(
                val.abs() < tol,
                "Element ({i},{j}) = {val} is not near zero (tol={tol})"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Cholesky decomposition
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cholesky_2x2() {
    let ctx = Context::new();
    // A = [[4, 2], [2, 3]]
    // Expected L = [[2, 0], [1, sqrt(2)]]
    let a = Matrix::new(vec![
        vec![ctx.int(4), ctx.int(2)],
        vec![ctx.int(2), ctx.int(3)],
    ])
    .unwrap();

    let l = a
        .cholesky()
        .expect("Cholesky should succeed for SPD matrix");

    // Verify L is lower triangular: L[0][1] should be 0
    let l01 = l.get(0, 1).simplify().eval_f64().unwrap();
    assert!(l01.abs() < 1e-10, "L[0][1] should be 0, got {l01}");

    // Verify L * Lᵀ = A
    let lt = l.transpose();
    let product = l.matmul(&lt).unwrap();
    assert_matrix_approx(&product, &[4.0, 2.0, 2.0, 3.0], 1e-9);
}

#[test]
fn cholesky_3x3() {
    let ctx = Context::new();
    // Classic 3×3 SPD example:
    // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
    // L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
    let a = Matrix::new(vec![
        vec![ctx.int(4), ctx.int(12), ctx.int(-16)],
        vec![ctx.int(12), ctx.int(37), ctx.int(-43)],
        vec![ctx.int(-16), ctx.int(-43), ctx.int(98)],
    ])
    .unwrap();

    let l = a
        .cholesky()
        .expect("Cholesky should succeed for SPD matrix");

    // Verify L * Lᵀ = A numerically
    let lt = l.transpose();
    let product = l.matmul(&lt).unwrap();
    assert_matrix_approx(
        &product,
        &[4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        1e-9,
    );

    // Verify specific L entries
    assert_matrix_approx(&l, &[2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0], 1e-9);
}

#[test]
fn cholesky_not_positive_definite() {
    let ctx = Context::new();
    // A = [[-1, 0], [0, 1]] — not positive definite (first diagonal is -1)
    let a = Matrix::new(vec![
        vec![ctx.int(-1), ctx.int(0)],
        vec![ctx.int(0), ctx.int(1)],
    ])
    .unwrap();

    assert!(
        a.cholesky().is_err(),
        "Cholesky should return Err for non-positive-definite matrix"
    );
}

#[test]
fn cholesky_identity() {
    // I₃ → L = I₃
    let ctx = Context::new();
    let eye = Matrix::identity(&ctx, 3);
    let l = eye.cholesky().expect("Cholesky of identity should succeed");

    // L should be the identity
    assert_matrix_approx(&l, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Pseudo-inverse
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pinv_full_rank() {
    let ctx = Context::new();
    // For a square invertible matrix, pinv = inv.
    // A = [[1, 2], [3, 4]], det = -2 ≠ 0
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3), ctx.int(4)],
    ])
    .unwrap();

    let pinv = a.pinv().expect("pinv should succeed for full-rank matrix");
    let inv = a.inv().expect("inv should succeed for invertible matrix");

    // pinv and inv should agree numerically
    let (nr, nc) = pinv.shape();
    assert_eq!((nr, nc), (2, 2));
    for i in 0..nr {
        for j in 0..nc {
            let pv = pinv.get(i, j).simplify().eval_f64().unwrap();
            let iv = inv.get(i, j).simplify().eval_f64().unwrap();
            assert!(
                common::approx_eq(pv, iv, 1e-9),
                "pinv[{i},{j}]={pv} != inv[{i},{j}]={iv}"
            );
        }
    }
}

#[test]
fn pinv_overdetermined() {
    let ctx = Context::new();
    // A is 3×2 (overdetermined, full column rank):
    // A = [[1, 0], [0, 1], [1, 1]]
    // A⁺A should equal I₂
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(0)],
        vec![ctx.int(0), ctx.int(1)],
        vec![ctx.int(1), ctx.int(1)],
    ])
    .unwrap();

    let pinv = a
        .pinv()
        .expect("pinv should succeed for full-column-rank matrix");

    // pinv should be 2×3
    assert_eq!(pinv.shape(), (2, 3));

    // A⁺ · A should be I₂
    let pinv_a = pinv.matmul(&a).unwrap();
    assert_matrix_approx(&pinv_a, &[1.0, 0.0, 0.0, 1.0], 1e-9);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Riccati residual
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn riccati_residual_setup() {
    let ctx = Context::new();
    // System: A = [[0, 1], [-2, -3]], B = [[0], [1]]
    // P = I₂, Q = I₂, R = [[1]]
    //
    // Residual = AᵀP + PA − PBR⁻¹BᵀP + Q
    //   AᵀP = [[0, -2], [1, -3]]
    //   PA  = [[0, 1], [-2, -3]]
    //   PBR⁻¹BᵀP = [[0,0],[0,1]]
    //   Q   = I₂
    //   Residual = [[1, -1], [-1, -6]]
    let a = Matrix::new(vec![
        vec![ctx.int(0), ctx.int(1)],
        vec![ctx.int(-2), ctx.int(-3)],
    ])
    .unwrap();
    let b = Matrix::new(vec![vec![ctx.int(0)], vec![ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![ctx.int(1), ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    let p = Matrix::identity(&ctx, 2);
    let q = Matrix::identity(&ctx, 2);
    let r = Matrix::new(vec![vec![ctx.int(1)]]).unwrap();

    let residual = ss
        .riccati_residual(&p, &q, &r)
        .expect("Riccati residual should succeed");

    assert_eq!(residual.shape(), (2, 2));
    assert_matrix_approx(&residual, &[1.0, -1.0, -1.0, -6.0], 1e-9);
}

#[test]
fn riccati_residual_at_solution() {
    let ctx = Context::new();
    // 1×1 system: A = [[0]], B = [[1]], Q = [[1]], R = [[1]]
    // CARE: AᵀP + PA - PBR⁻¹BᵀP + Q = 0
    //       0 + 0 - P² + 1 = 0  →  P = 1
    // With P = [[1]], residual should be zero.
    let a = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();
    let b = Matrix::new(vec![vec![ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![ctx.int(1)]]).unwrap();
    let d = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    let p = Matrix::new(vec![vec![ctx.int(1)]]).unwrap();
    let q = Matrix::new(vec![vec![ctx.int(1)]]).unwrap();
    let r = Matrix::new(vec![vec![ctx.int(1)]]).unwrap();

    let residual = ss
        .riccati_residual(&p, &q, &r)
        .expect("Riccati residual should succeed");

    assert_matrix_near_zero(&residual, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Ackermann pole placement
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ackermann_simple() {
    let ctx = Context::new();
    // Double integrator: A = [[0,1],[0,0]], B = [[0],[1]]
    // Desired poles: -1, -2
    // Expected K = [2, 3]
    //
    // Verification: A - BK = [[0,1],[-2,-3]]
    // char poly = s² + 3s + 2 = (s+1)(s+2) → eigenvalues -1, -2 ✓
    let a = Matrix::new(vec![
        vec![ctx.int(0), ctx.int(1)],
        vec![ctx.int(0), ctx.int(0)],
    ])
    .unwrap();
    let b = Matrix::new(vec![vec![ctx.int(0)], vec![ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![ctx.int(1), ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a.clone(), b.clone(), c, d);

    let desired_poles = vec![ctx.int(-1), ctx.int(-2)];
    let k = ss
        .ackermann(&desired_poles)
        .expect("Ackermann should succeed for controllable SISO system");

    // K should be 1×2
    assert_eq!(k.shape(), (1, 2));

    // Check K ≈ [2, 3]
    assert_matrix_approx(&k, &[2.0, 3.0], 1e-9);

    // Verify closed-loop eigenvalues of (A - BK)
    let bk = b.matmul(&k).unwrap();
    let a_cl = a.sub(&bk).unwrap();
    let mut eigs = a_cl.eigenvals().unwrap();
    eigs.sort_by(|a, b| {
        let va = a.eval_f64().unwrap_or(f64::NAN);
        let vb = b.eval_f64().unwrap_or(f64::NAN);
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    });
    assert_eq!(eigs.len(), 2, "Expected 2 eigenvalues");
    let e0 = eigs[0].eval_f64().unwrap();
    let e1 = eigs[1].eval_f64().unwrap();
    assert!(
        common::approx_eq(e0, -2.0, 1e-9),
        "First eigenvalue should be -2, got {e0}"
    );
    assert!(
        common::approx_eq(e1, -1.0, 1e-9),
        "Second eigenvalue should be -1, got {e1}"
    );
}

#[test]
fn ackermann_not_controllable_returns_err() {
    let ctx = Context::new();
    // A = [[1, 0], [0, 2]], B = [[1], [0]]
    // Controllability matrix C = [[1, 1], [0, 0]] → rank 1, not controllable
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(0)],
        vec![ctx.int(0), ctx.int(2)],
    ])
    .unwrap();
    let b = Matrix::new(vec![vec![ctx.int(1)], vec![ctx.int(0)]]).unwrap();
    let c = Matrix::new(vec![vec![ctx.int(1), ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    let desired_poles = vec![ctx.int(-1), ctx.int(-2)];
    assert!(
        ss.ackermann(&desired_poles).is_err(),
        "Ackermann should fail for an uncontrollable system"
    );
}

#[test]
fn ackermann_wrong_pole_count_returns_err() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(0), ctx.int(1)],
        vec![ctx.int(0), ctx.int(0)],
    ])
    .unwrap();
    let b = Matrix::new(vec![vec![ctx.int(0)], vec![ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![ctx.int(1), ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    let err = ss.ackermann(&[ctx.int(-1)]).unwrap_err();
    assert!(
        matches!(err, SymplexError::InvalidArgument { .. }),
        "one pole for a 2-state system must be rejected, got {err}"
    );
}

#[test]
fn ackermann_multi_input_returns_err() {
    let ctx = Context::new();
    // A = [[0, 1], [0, 0]], B = [[1, 0], [0, 1]] (2 inputs)
    let a = Matrix::new(vec![
        vec![ctx.int(0), ctx.int(1)],
        vec![ctx.int(0), ctx.int(0)],
    ])
    .unwrap();
    let b = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(0)],
        vec![ctx.int(0), ctx.int(1)],
    ])
    .unwrap();
    let c = Matrix::identity(&ctx, 2);
    let d = Matrix::zeros(&ctx, 2, 2);
    let ss = StateSpace::new(a, b, c, d);

    let desired_poles = vec![ctx.int(-1), ctx.int(-2)];
    assert!(
        ss.ackermann(&desired_poles).is_err(),
        "Ackermann should fail for a multi-input system"
    );
}
