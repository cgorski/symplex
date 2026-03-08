//! Integration tests for the quaternion algebra module.

use symplex::quaternion::Quaternion;

/// Helper: assert a float is close to an expected value.
fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{msg}: expected {expected}, got {actual}"
    );
}

/// Helper: evaluate a quaternion's components to f64 values.
fn quat_to_f64(q: &Quaternion) -> (f64, f64, f64, f64) {
    (
        q.w.eval().eval_f64().unwrap(),
        q.x.eval().eval_f64().unwrap(),
        q.y.eval().eval_f64().unwrap(),
        q.z.eval().eval_f64().unwrap(),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Identity quaternion
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_identity() {
    let id = Quaternion::identity();
    let (w, x, y, z) = quat_to_f64(&id);
    assert_close(w, 1.0, 1e-12, "identity w");
    assert_close(x, 0.0, 1e-12, "identity x");
    assert_close(y, 0.0, 1e-12, "identity y");
    assert_close(z, 0.0, 1e-12, "identity z");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Multiplication by identity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_mul_identity() {
    let q = Quaternion::new(
        symplex::default_context().int(1),
        symplex::default_context().int(2),
        symplex::default_context().int(3),
        symplex::default_context().int(4),
    );
    let id = Quaternion::identity();

    // q * identity = q
    let result = q.mul(&id);
    let (w, x, y, z) = quat_to_f64(&result.eval());
    assert_close(w, 1.0, 1e-12, "q*id w");
    assert_close(x, 2.0, 1e-12, "q*id x");
    assert_close(y, 3.0, 1e-12, "q*id y");
    assert_close(z, 4.0, 1e-12, "q*id z");

    // identity * q = q
    let result2 = id.mul(&q);
    let (w2, x2, y2, z2) = quat_to_f64(&result2.eval());
    assert_close(w2, 1.0, 1e-12, "id*q w");
    assert_close(x2, 2.0, 1e-12, "id*q x");
    assert_close(y2, 3.0, 1e-12, "id*q y");
    assert_close(z2, 4.0, 1e-12, "id*q z");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. q * conjugate(q) = |q|² (scalar quaternion)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_mul_conjugate() {
    let q = Quaternion::new(
        symplex::default_context().int(1),
        symplex::default_context().int(2),
        symplex::default_context().int(3),
        symplex::default_context().int(4),
    );
    let qc = q.conjugate();
    let product = q.mul(&qc).eval();

    let (w, x, y, z) = quat_to_f64(&product);
    // |q|² = 1 + 4 + 9 + 16 = 30
    assert_close(w, 30.0, 1e-12, "q*q* w should be |q|²");
    assert_close(x, 0.0, 1e-12, "q*q* x should be 0");
    assert_close(y, 0.0, 1e-12, "q*q* y should be 0");
    assert_close(z, 0.0, 1e-12, "q*q* z should be 0");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. i² = -1
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_i_squared() {
    let qi = Quaternion::new(
        symplex::default_context().int(0),
        symplex::default_context().int(1),
        symplex::default_context().int(0),
        symplex::default_context().int(0),
    );
    let result = qi.mul(&qi).eval();
    let (w, x, y, z) = quat_to_f64(&result);
    assert_close(w, -1.0, 1e-12, "i² w");
    assert_close(x, 0.0, 1e-12, "i² x");
    assert_close(y, 0.0, 1e-12, "i² y");
    assert_close(z, 0.0, 1e-12, "i² z");
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. j² = -1
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_j_squared() {
    let qj = Quaternion::new(
        symplex::default_context().int(0),
        symplex::default_context().int(0),
        symplex::default_context().int(1),
        symplex::default_context().int(0),
    );
    let result = qj.mul(&qj).eval();
    let (w, x, y, z) = quat_to_f64(&result);
    assert_close(w, -1.0, 1e-12, "j² w");
    assert_close(x, 0.0, 1e-12, "j² x");
    assert_close(y, 0.0, 1e-12, "j² y");
    assert_close(z, 0.0, 1e-12, "j² z");
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. k² = -1
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_k_squared() {
    let qk = Quaternion::new(
        symplex::default_context().int(0),
        symplex::default_context().int(0),
        symplex::default_context().int(0),
        symplex::default_context().int(1),
    );
    let result = qk.mul(&qk).eval();
    let (w, x, y, z) = quat_to_f64(&result);
    assert_close(w, -1.0, 1e-12, "k² w");
    assert_close(x, 0.0, 1e-12, "k² x");
    assert_close(y, 0.0, 1e-12, "k² y");
    assert_close(z, 0.0, 1e-12, "k² z");
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. i * j = k
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_ij_equals_k() {
    let qi = Quaternion::new(
        symplex::default_context().int(0),
        symplex::default_context().int(1),
        symplex::default_context().int(0),
        symplex::default_context().int(0),
    );
    let qj = Quaternion::new(
        symplex::default_context().int(0),
        symplex::default_context().int(0),
        symplex::default_context().int(1),
        symplex::default_context().int(0),
    );
    let result = qi.mul(&qj).eval();
    let (w, x, y, z) = quat_to_f64(&result);
    assert_close(w, 0.0, 1e-12, "i*j w");
    assert_close(x, 0.0, 1e-12, "i*j x");
    assert_close(y, 0.0, 1e-12, "i*j y");
    assert_close(z, 1.0, 1e-12, "i*j z (should be k)");
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. j * i = -k (non-commutative!)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_ji_equals_neg_k() {
    let qi = Quaternion::new(
        symplex::default_context().int(0),
        symplex::default_context().int(1),
        symplex::default_context().int(0),
        symplex::default_context().int(0),
    );
    let qj = Quaternion::new(
        symplex::default_context().int(0),
        symplex::default_context().int(0),
        symplex::default_context().int(1),
        symplex::default_context().int(0),
    );
    let result = qj.mul(&qi).eval();
    let (w, x, y, z) = quat_to_f64(&result);
    assert_close(w, 0.0, 1e-12, "j*i w");
    assert_close(x, 0.0, 1e-12, "j*i x");
    assert_close(y, 0.0, 1e-12, "j*i y");
    assert_close(z, -1.0, 1e-12, "j*i z (should be -k)");
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Conjugate verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_conjugate() {
    let q = Quaternion::new(
        symplex::default_context().int(5),
        symplex::default_context().int(3),
        symplex::default_context().int(-7),
        symplex::default_context().int(2),
    );
    let qc = q.conjugate().eval();
    let (w, x, y, z) = quat_to_f64(&qc);
    assert_close(w, 5.0, 1e-12, "conjugate w");
    assert_close(x, -3.0, 1e-12, "conjugate x");
    assert_close(y, 7.0, 1e-12, "conjugate y");
    assert_close(z, -2.0, 1e-12, "conjugate z");
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Norm squared: (1,2,3,4) → 1+4+9+16 = 30
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_norm_squared() {
    let q = Quaternion::new(
        symplex::default_context().int(1),
        symplex::default_context().int(2),
        symplex::default_context().int(3),
        symplex::default_context().int(4),
    );
    let n2 = q.norm_squared().eval().eval_f64().unwrap();
    assert_close(n2, 30.0, 1e-12, "|q|²");
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. q * q⁻¹ = identity (numerically)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_inverse() {
    let q = Quaternion::new(
        symplex::default_context().int(1),
        symplex::default_context().int(2),
        symplex::default_context().int(3),
        symplex::default_context().int(4),
    );
    let qi = q.inverse();
    let product = q.mul(&qi).eval();
    let (w, x, y, z) = quat_to_f64(&product);
    assert_close(w, 1.0, 1e-10, "q*q⁻¹ w");
    assert_close(x, 0.0, 1e-10, "q*q⁻¹ x");
    assert_close(y, 0.0, 1e-10, "q*q⁻¹ y");
    assert_close(z, 0.0, 1e-10, "q*q⁻¹ z");
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. Identity quaternion → I₃ rotation matrix
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_to_rotation_identity() {
    let id = Quaternion::identity();
    let r = id.to_rotation_matrix();

    assert_eq!(r.nrows(), 3);
    assert_eq!(r.ncols(), 3);

    for i in 0..3 {
        for j in 0..3 {
            let val = r.get(i, j).eval().eval_f64().unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(
                val,
                expected,
                1e-12,
                &format!("R_id[{i}][{j}]"),
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. (0,0,0,1) → rotation by π about z-axis
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_to_rotation_180_z() {
    // q = (0, 0, 0, 1) corresponds to 180° rotation about z-axis
    // Expected rotation matrix:
    // | -1  0  0 |
    // |  0 -1  0 |
    // |  0  0  1 |
    let q = Quaternion::new(
        symplex::default_context().int(0),
        symplex::default_context().int(0),
        symplex::default_context().int(0),
        symplex::default_context().int(1),
    );
    let r = q.to_rotation_matrix();

    let expected = [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    for (i, expected_row) in expected.iter().enumerate() {
        for (j, &exp_val) in expected_row.iter().enumerate() {
            let val = r.get(i, j).eval().eval_f64().unwrap();
            assert_close(
                val,
                exp_val,
                1e-12,
                &format!("R_180z[{i}][{j}]"),
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. from_axis_angle: 90° about z → verify rotation matrix
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_from_axis_angle_z_90() {
    let zero = symplex::default_context().int(0);
    let one = symplex::default_context().int(1);
    let angle = &symplex::default_context().pi() / &symplex::default_context().int(2); // π/2

    let q = Quaternion::from_axis_angle(&zero, &zero, &one, &angle);

    // q should be (cos(π/4), 0, 0, sin(π/4)) = (√2/2, 0, 0, √2/2)
    let (w, x, y, z) = quat_to_f64(&q.eval());
    let sqrt2_2 = std::f64::consts::FRAC_1_SQRT_2;
    assert_close(w, sqrt2_2, 1e-12, "axis-angle w");
    assert_close(x, 0.0, 1e-12, "axis-angle x");
    assert_close(y, 0.0, 1e-12, "axis-angle y");
    assert_close(z, sqrt2_2, 1e-12, "axis-angle z");

    // The rotation matrix for 90° about z should be:
    // | 0 -1  0 |
    // | 1  0  0 |
    // | 0  0  1 |
    let r = q.to_rotation_matrix();
    let expected = [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    for (i, expected_row) in expected.iter().enumerate() {
        for (j, &exp_val) in expected_row.iter().enumerate() {
            let val = r.get(i, j).eval().eval_f64().unwrap();
            assert_close(
                val,
                exp_val,
                1e-10,
                &format!("R_z90[{i}][{j}]"),
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. Angular velocity derivative structure
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_angular_velocity_derivative() {
    // For identity quaternion q = (1,0,0,0) and ω = (0,0,ωz):
    // ω_quat = (0, 0, 0, ωz)
    // q ⊗ ω_quat = (1,0,0,0)⊗(0,0,0,ωz)
    //   w = 1*0 - 0*0 - 0*0 - 0*ωz = 0
    //   x = 1*0 + 0*0 + 0*ωz - 0*0 = 0
    //   y = 1*0 - 0*ωz + 0*0 + 0*0 = 0
    //   z = 1*ωz + 0*0 - 0*0 + 0*0 = ωz
    // q̇ = ½ * (0, 0, 0, ωz)
    let q = Quaternion::identity();
    let zero = symplex::default_context().int(0);
    let wz = symplex::default_context().int(1); // ωz = 1

    let qdot = q.angular_velocity_derivative(&zero, &zero, &wz);
    let (w, x, y, z) = quat_to_f64(&qdot.eval());

    assert_close(w, 0.0, 1e-12, "q̇ w");
    assert_close(x, 0.0, 1e-12, "q̇ x");
    assert_close(y, 0.0, 1e-12, "q̇ y");
    assert_close(z, 0.5, 1e-12, "q̇ z (½ωz)");
}

// ═══════════════════════════════════════════════════════════════════════════
// 16. Display format
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_display() {
    let q = Quaternion::new(
        symplex::default_context().int(1),
        symplex::default_context().int(2),
        symplex::default_context().int(3),
        symplex::default_context().int(4),
    );
    let s = format!("{q}");
    assert_eq!(s, "(1 + 2i + 3j + 4k)");
}

// ═══════════════════════════════════════════════════════════════════════════
// 17. Zero quaternion
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_zero() {
    let z = Quaternion::zero();
    let (w, x, y, z) = quat_to_f64(&z);
    assert_close(w, 0.0, 1e-12, "zero w");
    assert_close(x, 0.0, 1e-12, "zero x");
    assert_close(y, 0.0, 1e-12, "zero y");
    assert_close(z, 0.0, 1e-12, "zero z");
}

// ═══════════════════════════════════════════════════════════════════════════
// 18. Pure quaternion from_vector
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_from_vector() {
    let vx = symplex::default_context().int(3);
    let vy = symplex::default_context().int(4);
    let vz = symplex::default_context().int(5);
    let q = Quaternion::from_vector(&vx, &vy, &vz);
    let (w, x, y, z) = quat_to_f64(&q);
    assert_close(w, 0.0, 1e-12, "from_vector w");
    assert_close(x, 3.0, 1e-12, "from_vector x");
    assert_close(y, 4.0, 1e-12, "from_vector y");
    assert_close(z, 5.0, 1e-12, "from_vector z");
}

// ═══════════════════════════════════════════════════════════════════════════
// 19. Hamilton product: ijk = -1
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_ijk_equals_neg_one() {
    let qi = Quaternion::new(
        symplex::default_context().int(0),
        symplex::default_context().int(1),
        symplex::default_context().int(0),
        symplex::default_context().int(0),
    );
    let qj = Quaternion::new(
        symplex::default_context().int(0),
        symplex::default_context().int(0),
        symplex::default_context().int(1),
        symplex::default_context().int(0),
    );
    let qk = Quaternion::new(
        symplex::default_context().int(0),
        symplex::default_context().int(0),
        symplex::default_context().int(0),
        symplex::default_context().int(1),
    );
    let ij = qi.mul(&qj);
    let ijk = ij.mul(&qk).eval();
    let (w, x, y, z) = quat_to_f64(&ijk);
    assert_close(w, -1.0, 1e-12, "ijk w");
    assert_close(x, 0.0, 1e-12, "ijk x");
    assert_close(y, 0.0, 1e-12, "ijk y");
    assert_close(z, 0.0, 1e-12, "ijk z");
}

// ═══════════════════════════════════════════════════════════════════════════
// 20. Normalize: verify unit norm
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_normalize() {
    let q = Quaternion::new(
        symplex::default_context().int(1),
        symplex::default_context().int(2),
        symplex::default_context().int(3),
        symplex::default_context().int(4),
    );
    let qn = q.normalize();
    let norm_val = qn.norm_squared().eval().eval_f64().unwrap();
    assert_close(norm_val, 1.0, 1e-10, "|normalize(q)|²");
}

// ═══════════════════════════════════════════════════════════════════════════
// 21. Substitution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_subs() {
    let theta = symplex::default_context().symbol("theta");
    let q = Quaternion::new(
        theta.cos(),
        theta.sin(),
        symplex::default_context().int(0),
        symplex::default_context().int(0),
    );
    let pi_half = &symplex::default_context().pi() / &symplex::default_context().int(2);
    let q2 = q.subs(&theta, &pi_half).eval();
    let (w, x, y, z) = quat_to_f64(&q2);
    assert_close(w, 0.0, 1e-12, "subs w = cos(π/2)");
    assert_close(x, 1.0, 1e-12, "subs x = sin(π/2)");
    assert_close(y, 0.0, 1e-12, "subs y");
    assert_close(z, 0.0, 1e-12, "subs z");
}

// ═══════════════════════════════════════════════════════════════════════════
// 22. Rotation matrix for 90° about x-axis
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_from_axis_angle_x_90() {
    let one = symplex::default_context().int(1);
    let zero = symplex::default_context().int(0);
    let angle = &symplex::default_context().pi() / &symplex::default_context().int(2);

    let q = Quaternion::from_axis_angle(&one, &zero, &zero, &angle);
    let r = q.to_rotation_matrix();

    // 90° about x:
    // | 1  0  0 |
    // | 0  0 -1 |
    // | 0  1  0 |
    let expected = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ];

    for (i, expected_row) in expected.iter().enumerate() {
        for (j, &exp_val) in expected_row.iter().enumerate() {
            let val = r.get(i, j).eval().eval_f64().unwrap();
            assert_close(
                val,
                exp_val,
                1e-10,
                &format!("R_x90[{i}][{j}]"),
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 23. Rotation matrix is orthogonal (R * Rᵀ = I) for a general quaternion
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_rotation_matrix_orthogonal() {
    // Use a unit quaternion built from axis-angle
    let angle = &symplex::default_context().pi() / &symplex::default_context().int(3); // 60°
    // Axis: (1, 1, 1)/√3
    let inv_sqrt3 = symplex::default_context().int(3).sqrt();
    let ax = &symplex::default_context().int(1) / &inv_sqrt3;
    let ay = &symplex::default_context().int(1) / &inv_sqrt3;
    let az = &symplex::default_context().int(1) / &inv_sqrt3;

    let q = Quaternion::from_axis_angle(&ax, &ay, &az, &angle);
    let r = q.to_rotation_matrix();
    let rt = r.transpose();
    let product = r.matmul(&rt).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            let val = product.get(i, j).eval().eval_f64().unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(
                val,
                expected,
                1e-10,
                &format!("R*Rᵀ[{i}][{j}]"),
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 24. Hamilton product associativity: (p*q)*r = p*(q*r)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_mul_associative() {
    let p = Quaternion::new(
        symplex::default_context().int(1),
        symplex::default_context().int(2),
        symplex::default_context().int(3),
        symplex::default_context().int(4),
    );
    let q = Quaternion::new(
        symplex::default_context().int(5),
        symplex::default_context().int(-1),
        symplex::default_context().int(2),
        symplex::default_context().int(-3),
    );
    let r = Quaternion::new(
        symplex::default_context().int(-2),
        symplex::default_context().int(1),
        symplex::default_context().int(0),
        symplex::default_context().int(7),
    );

    let lhs = p.mul(&q).mul(&r).eval();
    let rhs = p.mul(&q.mul(&r)).eval();

    let (lw, lx, ly, lz) = quat_to_f64(&lhs);
    let (rw, rx, ry, rz) = quat_to_f64(&rhs);

    assert_close(lw, rw, 1e-10, "associativity w");
    assert_close(lx, rx, 1e-10, "associativity x");
    assert_close(ly, ry, 1e-10, "associativity y");
    assert_close(lz, rz, 1e-10, "associativity z");
}

// ═══════════════════════════════════════════════════════════════════════════
// 25. Norm of a product: |p*q| = |p|*|q|
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_norm_product() {
    let p = Quaternion::new(
        symplex::default_context().int(1),
        symplex::default_context().int(2),
        symplex::default_context().int(3),
        symplex::default_context().int(4),
    );
    let q = Quaternion::new(
        symplex::default_context().int(5),
        symplex::default_context().int(-1),
        symplex::default_context().int(2),
        symplex::default_context().int(-3),
    );

    let norm_p = p.norm().eval().eval_f64().unwrap();
    let norm_q = q.norm().eval().eval_f64().unwrap();
    let norm_pq = p.mul(&q).norm().eval().eval_f64().unwrap();

    assert_close(norm_pq, norm_p * norm_q, 1e-10, "|p*q| = |p|*|q|");
}
