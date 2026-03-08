//! Tests for Wave D: matrix inverse, characteristic polynomial, and eigenvalues.

use symplex::matrix::Matrix;

// ═══════════════════════════════════════════════════════════════════════════
// Minor
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn minor_2x2_removes_row_and_col() {
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(3), symplex::int(4)],
    ]).unwrap();
    // Removing row 0, col 0 → [[4]]
    let m00 = m.minor(0, 0).unwrap();
    assert_eq!(m00.nrows(), 1);
    assert_eq!(m00.ncols(), 1);
    assert_eq!(format!("{}", m00.get(0, 0)), "4");

    // Removing row 0, col 1 → [[3]]
    let m01 = m.minor(0, 1).unwrap();
    assert_eq!(format!("{}", m01.get(0, 0)), "3");

    // Removing row 1, col 0 → [[2]]
    let m10 = m.minor(1, 0).unwrap();
    assert_eq!(format!("{}", m10.get(0, 0)), "2");

    // Removing row 1, col 1 → [[1]]
    let m11 = m.minor(1, 1).unwrap();
    assert_eq!(format!("{}", m11.get(0, 0)), "1");
}

#[test]
fn minor_3x3_produces_2x2() {
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2), symplex::int(3)],
        vec![symplex::int(4), symplex::int(5), symplex::int(6)],
        vec![symplex::int(7), symplex::int(8), symplex::int(9)],
    ]).unwrap();
    // Remove row 1, col 1 → [[1,3],[7,9]]
    let sub = m.minor(1, 1).unwrap();
    assert_eq!(sub.nrows(), 2);
    assert_eq!(sub.ncols(), 2);
    assert_eq!(format!("{}", sub.get(0, 0)), "1");
    assert_eq!(format!("{}", sub.get(0, 1)), "3");
    assert_eq!(format!("{}", sub.get(1, 0)), "7");
    assert_eq!(format!("{}", sub.get(1, 1)), "9");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cofactor
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cofactor_2x2_numeric() {
    let m = Matrix::new(vec![
        vec![symplex::int(3), symplex::int(7)],
        vec![symplex::int(1), symplex::int(5)],
    ]).unwrap();
    // C(0,0) = (+1)*det([[5]]) = 5
    let c00 = m.cofactor(0, 0).unwrap();
    assert_eq!(format!("{c00}"), "5");
    // C(0,1) = (-1)*det([[1]]) = -1
    let c01 = m.cofactor(0, 1).unwrap();
    assert_eq!(format!("{c01}"), "-1");
    // C(1,0) = (-1)*det([[7]]) = -7
    let c10 = m.cofactor(1, 0).unwrap();
    assert_eq!(format!("{c10}"), "-7");
    // C(1,1) = (+1)*det([[3]]) = 3
    let c11 = m.cofactor(1, 1).unwrap();
    assert_eq!(format!("{c11}"), "3");
}

// ═══════════════════════════════════════════════════════════════════════════
// Adjugate
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn adjugate_2x2_numeric() {
    // For [[a,b],[c,d]], adj = [[d,-b],[-c,a]]
    let m = Matrix::new(vec![
        vec![symplex::int(3), symplex::int(7)],
        vec![symplex::int(1), symplex::int(5)],
    ]).unwrap();
    let adj = m.adjugate().unwrap();
    assert_eq!(format!("{}", adj.get(0, 0)), "5");
    assert_eq!(format!("{}", adj.get(0, 1)), "-7");
    assert_eq!(format!("{}", adj.get(1, 0)), "-1");
    assert_eq!(format!("{}", adj.get(1, 1)), "3");
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn inverse_identity_is_identity() {
    for n in 1..=4 {
        let id = Matrix::identity(n);
        let inv = id.inv().expect("identity should be invertible");
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { "1" } else { "0" };
                let got = format!("{}", inv.get(i, j).simplify());
                assert_eq!(
                    got, expected,
                    "inv(I{n})[{i},{j}]: expected {expected}, got {got}"
                );
            }
        }
    }
}

#[test]
fn inverse_2x2_numeric() {
    // [[3,7],[1,5]], det=8
    // inv = (1/8)*[[5,-7],[-1,3]]
    let m = Matrix::new(vec![
        vec![symplex::int(3), symplex::int(7)],
        vec![symplex::int(1), symplex::int(5)],
    ]).unwrap();
    let inv = m.inv().expect("non-singular 2x2 should be invertible");
    let inv = inv.simplify();
    assert_eq!(format!("{}", inv.get(0, 0)), "5/8");
    assert_eq!(format!("{}", inv.get(0, 1)), "-7/8");
    assert_eq!(format!("{}", inv.get(1, 0)), "-1/8");
    assert_eq!(format!("{}", inv.get(1, 1)), "3/8");
}

#[test]
fn inverse_singular_returns_none() {
    // [[1,2],[2,4]] has det=0
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(2), symplex::int(4)],
    ]).unwrap();
    assert!(m.inv().is_err(), "singular matrix should return Err");
}

#[test]
fn inverse_times_original_is_identity_2x2() {
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(3), symplex::int(4)],
    ]).unwrap();
    let inv = m.inv().expect("non-singular");
    let product = m.matmul(&inv).unwrap().simplify();
    for i in 0..2 {
        for j in 0..2 {
            let val = format!("{}", product.get(i, j));
            let expected = if i == j { "1" } else { "0" };
            assert_eq!(
                val, expected,
                "M*M^-1 [{i},{j}]: expected {expected}, got {val}"
            );
        }
    }
}

#[test]
fn inverse_3x3_numeric() {
    // Upper-triangular [[1,2,3],[0,1,4],[0,0,1]], det = 1
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2), symplex::int(3)],
        vec![symplex::int(0), symplex::int(1), symplex::int(4)],
        vec![symplex::int(0), symplex::int(0), symplex::int(1)],
    ]).unwrap();
    let det = m.det().unwrap();
    assert_eq!(format!("{det}"), "1", "det should be 1");

    let inv = m.inv().expect("det=1, should be invertible");
    // For det=1, inv = adj(A). Verify M * M^-1 = I.
    let product = m.matmul(&inv).unwrap().simplify();
    for i in 0..3 {
        for j in 0..3 {
            let val = format!("{}", product.get(i, j));
            let expected = if i == j { "1" } else { "0" };
            assert_eq!(
                val, expected,
                "M*M^-1 [{i},{j}]: expected {expected}, got {val}"
            );
        }
    }
}

#[test]
fn inverse_1x1() {
    let m = Matrix::new(vec![vec![symplex::int(5)]]).unwrap();
    let inv = m
        .inv()
        .expect("1x1 with nonzero entry should be invertible");
    assert_eq!(format!("{}", inv.get(0, 0).simplify()), "1/5");
}

// ═══════════════════════════════════════════════════════════════════════════
// Characteristic polynomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn char_poly_identity_2x2() {
    // det(I - λI) = det([1-λ, 0; 0, 1-λ]) = (1-λ)² = λ² - 2λ + 1
    let lambda = symplex::var("lambda");
    let id = Matrix::identity(2);
    let cp = id.char_poly(&lambda).unwrap();
    let s = format!("{cp}");
    // Should contain lambda^2
    assert!(
        s.contains("lambda"),
        "char poly of I2 should contain lambda: {s}"
    );
    // Verify that substituting λ=1 gives 0
    let at_one = cp.subs(&lambda, &symplex::int(1)).simplify();
    assert_eq!(
        format!("{at_one}"),
        "0",
        "char poly of I2 at lambda=1 should be 0"
    );
}

#[test]
fn char_poly_2x2_numeric() {
    // [[2,1],[1,2]]
    // det([[2-λ, 1],[1, 2-λ]]) = (2-λ)² - 1 = λ²-4λ+3
    let lambda = symplex::var("lambda");
    let m = Matrix::new(vec![
        vec![symplex::int(2), symplex::int(1)],
        vec![symplex::int(1), symplex::int(2)],
    ]).unwrap();
    let cp = m.char_poly(&lambda).unwrap();
    let s = format!("{cp}");
    assert!(s.contains("lambda"), "char poly should mention lambda: {s}");

    // Eigenvalues are 1 and 3: verify cp(1)=0 and cp(3)=0
    let at_1 = cp.subs(&lambda, &symplex::int(1)).simplify();
    assert_eq!(format!("{at_1}"), "0", "cp(1) should be 0, got: {at_1}");

    let at_3 = cp.subs(&lambda, &symplex::int(3)).simplify();
    assert_eq!(format!("{at_3}"), "0", "cp(3) should be 0, got: {at_3}");
}

#[test]
fn char_poly_diagonal_3x3() {
    // diag(2, 5, 7) → cp = (2-λ)(5-λ)(7-λ)
    let lambda = symplex::var("lambda");
    let m = Matrix::new(vec![
        vec![symplex::int(2), symplex::int(0), symplex::int(0)],
        vec![symplex::int(0), symplex::int(5), symplex::int(0)],
        vec![symplex::int(0), symplex::int(0), symplex::int(7)],
    ]).unwrap();
    let cp = m.char_poly(&lambda).unwrap();

    // Verify roots at 2, 5, 7
    for val in [2, 5, 7] {
        let at_val = cp.subs(&lambda, &symplex::int(val)).simplify();
        assert_eq!(
            format!("{at_val}"),
            "0",
            "cp({val}) should be 0, got: {at_val}"
        );
    }
}

#[test]
fn char_poly_1x1() {
    let lambda = symplex::var("lambda");
    let m = Matrix::new(vec![vec![symplex::int(42)]]).unwrap();
    let cp = m.char_poly(&lambda).unwrap();
    // det([[42 - λ]]) = 42 - λ
    // Substituting λ=42 → 0
    let at_42 = cp.subs(&lambda, &symplex::int(42)).simplify();
    assert_eq!(format!("{at_42}"), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Eigenvalues
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eigenvals_2x2() {
    // [[2,1],[1,2]] → eigenvalues 1, 3
    let lambda = symplex::var("lambda");
    let m = Matrix::new(vec![
        vec![symplex::int(2), symplex::int(1)],
        vec![symplex::int(1), symplex::int(2)],
    ]).unwrap();
    let evals = m.eigenvals(&lambda).unwrap();
    assert_eq!(
        evals.len(),
        2,
        "2x2 should have 2 eigenvalues, got {}",
        evals.len()
    );

    let mut vals: Vec<String> = evals.iter().map(|e| format!("{e}")).collect();
    vals.sort();
    assert_eq!(
        vals,
        vec!["1", "3"],
        "eigenvalues should be 1 and 3, got: {vals:?}"
    );
}

#[test]
fn eigenvals_3x3_diagonal() {
    // diag(1, 2, 3) → eigenvalues 1, 2, 3
    let lambda = symplex::var("lambda");
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(0), symplex::int(0)],
        vec![symplex::int(0), symplex::int(2), symplex::int(0)],
        vec![symplex::int(0), symplex::int(0), symplex::int(3)],
    ]).unwrap();
    let evals = m.eigenvals(&lambda).unwrap();
    assert_eq!(
        evals.len(),
        3,
        "3x3 diagonal should have 3 eigenvalues, got {}",
        evals.len()
    );

    let mut vals: Vec<String> = evals.iter().map(|e| format!("{e}")).collect();
    vals.sort();
    assert_eq!(
        vals,
        vec!["1", "2", "3"],
        "eigenvalues should be 1, 2, 3, got: {vals:?}"
    );
}

#[test]
fn eigenvals_identity() {
    // I_2 → eigenvalue 1 (double)
    let lambda = symplex::var("lambda");
    let id = Matrix::identity(2);
    let evals = id.eigenvals(&lambda).unwrap();
    // The solver may return [1, 1] or just [1] depending on multiplicity handling.
    assert!(
        !evals.is_empty(),
        "identity should have at least one eigenvalue"
    );
    for e in &evals {
        assert_eq!(
            format!("{e}"),
            "1",
            "identity eigenvalue should be 1, got: {e}"
        );
    }
}

#[test]
fn eigenvals_1x1() {
    let lambda = symplex::var("lambda");
    let m = Matrix::new(vec![vec![symplex::int(7)]]).unwrap();
    let evals = m.eigenvals(&lambda).unwrap();
    assert_eq!(evals.len(), 1, "1x1 should have 1 eigenvalue");
    assert_eq!(format!("{}", evals[0]), "7");
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge cases & panics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[should_panic(expected = "requires a square matrix")]
fn minor_non_square_panics() {
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2), symplex::int(3)],
        vec![symplex::int(4), symplex::int(5), symplex::int(6)],
    ]).unwrap();
    let _ = m.minor(0, 0).unwrap();
}

#[test]
#[should_panic(expected = "requires a square matrix")]
fn inverse_non_square_panics() {
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2), symplex::int(3)],
        vec![symplex::int(4), symplex::int(5), symplex::int(6)],
    ]).unwrap();
    let _ = m.inv().unwrap();
}

#[test]
#[should_panic(expected = "requires a square matrix")]
fn char_poly_non_square_panics() {
    let lambda = symplex::var("lambda");
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(3), symplex::int(4)],
        vec![symplex::int(5), symplex::int(6)],
    ]).unwrap();
    let _ = m.char_poly(&lambda).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// Eigenvector verification: Av = λv
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eigenvector_satisfies_eigenvalue_equation() {
    // Matrix [[2,1],[1,2]] has eigenvalues 1 and 3.
    // Eigenvector for λ=1: [1, -1]  (A*v = 1*v)
    // Eigenvector for λ=3: [1,  1]  (A*v = 3*v)
    let a = Matrix::new(vec![
        vec![symplex::int(2), symplex::int(1)],
        vec![symplex::int(1), symplex::int(2)],
    ]).unwrap();

    // Verify eigenvalues first
    let lambda = symplex::var("lambda");
    let evals = a.eigenvals(&lambda).unwrap();
    assert_eq!(evals.len(), 2, "should have 2 eigenvalues");

    // Eigenvector for λ=1: v = [1, -1]
    let v1 = Matrix::col_vector(vec![symplex::int(1), symplex::int(-1)]);
    let av1 = a.matmul(&v1).unwrap().simplify();
    let lv1 = v1.scale(&symplex::int(1)).simplify(); // 1 * v1
    for i in 0..2 {
        let diff = (av1.get(i, 0) - lv1.get(i, 0)).simplify();
        assert!(
            diff.is_zero_structural(),
            "Av != λv for λ=1 at row {i}: Av={}, λv={}",
            av1.get(i, 0),
            lv1.get(i, 0)
        );
    }

    // Eigenvector for λ=3: v = [1, 1]
    let v3 = Matrix::col_vector(vec![symplex::int(1), symplex::int(1)]);
    let av3 = a.matmul(&v3).unwrap().simplify();
    let lv3 = v3.scale(&symplex::int(3)).simplify(); // 3 * v3
    for i in 0..2 {
        let diff = (av3.get(i, 0) - lv3.get(i, 0)).simplify();
        assert!(
            diff.is_zero_structural(),
            "Av != λv for λ=3 at row {i}: Av={}, λv={}",
            av3.get(i, 0),
            lv3.get(i, 0)
        );
    }
}

#[test]
fn complex_eigenvalues_rotation_matrix() {
    // Rotation matrix [[0, -1], [1, 0]] has eigenvalues ±i.
    // Characteristic polynomial: λ² + 1 = 0 → no real roots.
    let lambda = symplex::var("lambda");
    let m = Matrix::new(vec![
        vec![symplex::int(0), symplex::int(-1)],
        vec![symplex::int(1), symplex::int(0)],
    ]).unwrap();

    // Verify characteristic polynomial: λ² + 1
    let cp = m.char_poly(&lambda).unwrap();
    // cp(0) should be 1 (det of original matrix)
    let cp_at_0 = cp.subs(&lambda, &symplex::int(0)).simplify();
    assert_eq!(
        format!("{cp_at_0}"),
        "1",
        "cp(0) should be det = 1, got: {cp_at_0}"
    );

    // The solver returns complex eigenvalues ±i
    let evals = m.eigenvals(&lambda).unwrap();
    assert_eq!(
        evals.len(),
        2,
        "rotation matrix should have 2 eigenvalues (complex), got: {:?}",
        evals.iter().map(|e| format!("{e}")).collect::<Vec<_>>()
    );
    let mut vals: Vec<String> = evals.iter().map(|e| format!("{e}")).collect();
    vals.sort();
    assert_eq!(
        vals,
        vec!["-I", "I"],
        "eigenvalues of rotation matrix should be ±i, got: {vals:?}"
    );

    // Verify that λ² + 1 has no real roots by checking it's always positive for real λ
    let cp_at_5 = cp.subs(&lambda, &symplex::int(5)).simplify();
    let val = cp_at_5.eval_f64().expect("cp(5) should evaluate");
    assert!(val > 0.0, "cp(5) = 5² + 1 = 26, got {val}");
}

// ═══════════════════════════════════════════════════════════════════════════
// det(A⁻¹) = 1/det(A) verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn det_inverse_equals_reciprocal_det() {
    // For A = [[3,7],[1,5]], det(A) = 8, so det(A⁻¹) should be 1/8
    let a = Matrix::new(vec![
        vec![symplex::int(3), symplex::int(7)],
        vec![symplex::int(1), symplex::int(5)],
    ]).unwrap();
    let det_a = a.det().unwrap();
    assert_eq!(format!("{det_a}"), "8", "det(A) should be 8");

    let inv_a = a.inv().expect("non-singular matrix should be invertible");
    let det_inv = inv_a.simplify().det().unwrap().simplify();
    assert_eq!(
        format!("{det_inv}"),
        "1/8",
        "det(A⁻¹) should be 1/det(A) = 1/8, got: {det_inv}"
    );

    // Also verify with a 3x3: upper-triangular [[1,2,3],[0,1,4],[0,0,1]], det=1
    let b = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2), symplex::int(3)],
        vec![symplex::int(0), symplex::int(1), symplex::int(4)],
        vec![symplex::int(0), symplex::int(0), symplex::int(1)],
    ]).unwrap();
    let det_b = b.det().unwrap();
    assert_eq!(format!("{det_b}"), "1", "det(B) should be 1");

    let inv_b = b.inv().expect("det=1 should be invertible");
    let det_inv_b = inv_b.simplify().det().unwrap().simplify();
    assert_eq!(
        format!("{det_inv_b}"),
        "1",
        "det(B⁻¹) should be 1/1 = 1, got: {det_inv_b}"
    );
}
