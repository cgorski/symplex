//! Tests for Wave E: matrix decompositions and utilities.

use symplex::matrix::{Matrix, cross, dot};
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// RREF
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rref_identity() {
    let ctx = Context::new();
    let m = Matrix::identity(&ctx, 3);
    let (rref_mat, pivots) = m.rref();
    assert_eq!(pivots, vec![0, 1, 2]);
    // RREF of identity is identity
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { "1" } else { "0" };
            assert_eq!(format!("{}", rref_mat.get(i, j)), expected);
        }
    }
}

#[test]
fn rref_2x3() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    let (rref_mat, pivots) = m.rref();
    assert_eq!(pivots.len(), 2); // rank 2
    // First pivot column 0, second pivot column 1
    assert_eq!(pivots, vec![0, 1]);
    // Leading 1s
    assert_eq!(format!("{}", rref_mat.get(0, 0)), "1");
    assert_eq!(format!("{}", rref_mat.get(1, 1)), "1");
    // Entry above second pivot is zero
    assert_eq!(format!("{}", rref_mat.get(0, 1)), "0");
}

#[test]
fn rref_rank_deficient() {
    let ctx = Context::new();
    // Two proportional rows: rank should be 1
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(1), ctx.int(2)],
    ])
    .unwrap();
    let (_, pivots) = m.rref();
    assert_eq!(pivots.len(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Rank
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rank_full() {
    let ctx = Context::new();
    let m = Matrix::identity(&ctx, 3);
    assert_eq!(m.rank(), 3);
}

#[test]
fn rank_of_zero_matrix() {
    let ctx = Context::new();
    let m = Matrix::zeros(&ctx, 3, 3);
    assert_eq!(m.rank(), 0);
}

#[test]
fn rank_rectangular() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(0), ctx.int(2)],
        vec![ctx.int(0), ctx.int(1), ctx.int(3)],
    ])
    .unwrap();
    assert_eq!(m.rank(), 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Nullspace
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nullspace_identity_empty() {
    let ctx = Context::new();
    let m = Matrix::identity(&ctx, 3);
    assert!(m.nullspace().is_empty());
}

#[test]
fn nullspace_rank_deficient() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(2), ctx.int(4)],
    ])
    .unwrap();
    let ns = m.nullspace();
    assert_eq!(ns.len(), 1, "rank-1 2x2 should have 1-dim nullspace");
    // The nullspace vector should be a 2x1 column vector
    assert_eq!(ns[0].nrows(), 2);
    assert_eq!(ns[0].ncols(), 1);
}

#[test]
fn nullspace_vector_is_in_kernel() {
    let ctx = Context::new();
    // A = [[1, 2], [2, 4]]  =>  nullspace basis includes [-2, 1]
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(2), ctx.int(4)],
    ])
    .unwrap();
    let ns = a.nullspace();
    assert_eq!(ns.len(), 1);
    // Multiply A * v; result should be the zero vector (structurally after simplify)
    let product = a.matmul(&ns[0]).unwrap().simplify();
    for i in 0..product.nrows() {
        assert!(
            product.get(i, 0).is_zero_structural(),
            "A*v row {} should be zero, got {}",
            i,
            product.get(i, 0)
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Column space
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn columnspace_identity() {
    let ctx = Context::new();
    let m = Matrix::identity(&ctx, 3);
    let cs = m.columnspace();
    assert_eq!(cs.len(), 3, "identity has full column rank");
    for v in &cs {
        assert_eq!(v.nrows(), 3);
        assert_eq!(v.ncols(), 1);
    }
}

#[test]
fn columnspace_rank_deficient() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(2), ctx.int(4)],
    ])
    .unwrap();
    let cs = m.columnspace();
    assert_eq!(cs.len(), 1, "rank-1 matrix has 1-dim column space");
}

// ═══════════════════════════════════════════════════════════════════════════
// LU decomposition
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lu_identity() {
    let ctx = Context::new();
    let m = Matrix::identity(&ctx, 3);
    let (l, u, perm) = m.lu().expect("identity should have LU");
    // L and U should both be identity for an identity input
    for (i, &p) in perm.iter().enumerate() {
        assert_eq!(p, i);
        for j in 0..3 {
            let expected = if i == j { "1" } else { "0" };
            assert_eq!(format!("{}", l.get(i, j)), expected, "L[{i},{j}]");
            assert_eq!(format!("{}", u.get(i, j)), expected, "U[{i},{j}]");
        }
    }
}

#[test]
fn lu_2x2_verify_pa_eq_lu() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(7)],
    ])
    .unwrap();
    let (l, u, perm) = a.lu().expect("non-singular 2x2 should have LU");

    // Reconstruct PA
    let pa = Matrix::new(
        perm.iter()
            .map(|&r| (0..a.ncols()).map(|c| a.get(r, c).clone()).collect())
            .collect(),
    )
    .unwrap();

    // Reconstruct LU
    let lu = l.matmul(&u).unwrap().simplify();

    // PA should equal LU
    for i in 0..2 {
        for j in 0..2 {
            let diff = (pa.get(i, j) - lu.get(i, j)).simplify();
            assert!(
                diff.is_zero_structural(),
                "PA != LU at ({i},{j}): PA={}, LU={}",
                pa.get(i, j),
                lu.get(i, j)
            );
        }
    }
}

#[test]
fn lu_singular_returns_none() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(2), ctx.int(4)],
    ])
    .unwrap();
    assert!(m.lu().is_none(), "singular matrix should return None");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross product
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cross_product_basic() {
    let ctx = Context::new();
    // i x j = k
    let i_vec = Matrix::col_vector(vec![ctx.int(1), ctx.int(0), ctx.int(0)]);
    let j_vec = Matrix::col_vector(vec![ctx.int(0), ctx.int(1), ctx.int(0)]);
    let k_vec = cross(&i_vec, &j_vec);
    assert_eq!(k_vec.nrows(), 3);
    assert_eq!(k_vec.ncols(), 1);
    assert_eq!(format!("{}", k_vec.get(0, 0)), "0"); // x-component
    assert_eq!(format!("{}", k_vec.get(1, 0)), "0"); // y-component
    assert_eq!(format!("{}", k_vec.get(2, 0)), "1"); // z-component
}

#[test]
fn cross_product_anticommutative() {
    let ctx = Context::new();
    let a = Matrix::col_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]);
    let b = Matrix::col_vector(vec![ctx.int(4), ctx.int(5), ctx.int(6)]);
    let ab = cross(&a, &b).simplify();
    let ba = cross(&b, &a).simplify();
    // a x b = -(b x a)
    for i in 0..3 {
        let sum = (ab.get(i, 0) + ba.get(i, 0)).simplify();
        assert!(
            sum.is_zero_structural(),
            "cross product not anticommutative at component {i}: axb={}, bxa={}",
            ab.get(i, 0),
            ba.get(i, 0)
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dot product
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dot_product_basic() {
    let ctx = Context::new();
    let a = Matrix::col_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]);
    let b = Matrix::col_vector(vec![ctx.int(4), ctx.int(5), ctx.int(6)]);
    let result = dot(&a, &b);
    assert_eq!(format!("{result}"), "32"); // 4+10+18
}

// ═══════════════════════════════════════════════════════════════════════════
// Norm
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn norm_identity() {
    let ctx = Context::new();
    let m = Matrix::identity(&ctx, 2);
    // Frobenius norm of 2x2 identity is sqrt(2)
    let n = m.norm();
    let v = n
        .eval_f64()
        .expect("norm of identity should evaluate to f64");
    assert!(
        (v - std::f64::consts::SQRT_2).abs() < 1e-10,
        "expected sqrt(2), got {v}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Utilities: is_square, is_symmetric, hstack, vstack
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn is_square_and_not() {
    let ctx = Context::new();
    assert!(Matrix::identity(&ctx, 3).is_square());
    let rect = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    assert!(!rect.is_square());
}

#[test]
fn is_symmetric_true() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(2), ctx.int(1)],
    ])
    .unwrap();
    assert!(m.is_symmetric());
}

#[test]
fn is_symmetric_false() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3), ctx.int(1)],
    ])
    .unwrap();
    assert!(!m.is_symmetric());
}

#[test]
fn hstack_two_matrices() {
    let ctx = Context::new();
    let a = Matrix::col_vector(vec![ctx.int(1), ctx.int(2)]);
    let b = Matrix::col_vector(vec![ctx.int(3), ctx.int(4)]);
    let h = Matrix::hstack(&[&a, &b]).unwrap();
    assert_eq!(h.nrows(), 2);
    assert_eq!(h.ncols(), 2);
    assert_eq!(format!("{}", h.get(0, 0)), "1");
    assert_eq!(format!("{}", h.get(0, 1)), "3");
    assert_eq!(format!("{}", h.get(1, 0)), "2");
    assert_eq!(format!("{}", h.get(1, 1)), "4");
}

#[test]
fn vstack_two_matrices() {
    let ctx = Context::new();
    let a = Matrix::row_vector(vec![ctx.int(1), ctx.int(2)]);
    let b = Matrix::row_vector(vec![ctx.int(3), ctx.int(4)]);
    let v = Matrix::vstack(&[&a, &b]).unwrap();
    assert_eq!(v.nrows(), 2);
    assert_eq!(v.ncols(), 2);
    assert_eq!(format!("{}", v.get(0, 0)), "1");
    assert_eq!(format!("{}", v.get(0, 1)), "2");
    assert_eq!(format!("{}", v.get(1, 0)), "3");
    assert_eq!(format!("{}", v.get(1, 1)), "4");
}

#[test]
fn rank_nullity_theorem() {
    let ctx = Context::new();
    // For an m x n matrix, rank + nullity = n
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
        vec![ctx.int(7), ctx.int(8), ctx.int(9)],
    ])
    .unwrap();
    let r = m.rank();
    let ns = m.nullspace();
    assert_eq!(
        r + ns.len(),
        m.ncols(),
        "rank({r}) + nullity({}) should equal ncols({})",
        ns.len(),
        m.ncols()
    );
}
