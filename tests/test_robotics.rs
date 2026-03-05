//! Integration tests for the robotics module: DH parameters, forward kinematics, Jacobian.

mod common;

use symplex::matrix::jacobian;
use symplex::robotics::*;

// ═══════════════════════════════════════════════════════════════════════════
// 1. DH matrix with identity parameters
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dh_matrix_identity_params() {
    // θ=0, d=0, a=0, α=0 → should be the 4×4 identity matrix
    let zero = symplex::int(0);
    let t = dh_matrix(&zero, &zero, &zero, &zero);

    assert_eq!(t.nrows(), 4);
    assert_eq!(t.ncols(), 4);

    // Evaluate every entry numerically and compare to identity
    for i in 0..4 {
        for j in 0..4 {
            let val = t.get(i, j).eval().evalf_f64().unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-12,
                "DH identity: entry ({i},{j}) = {val}, expected {expected}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. DH matrix with pure rotation (θ=π/2)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dh_matrix_pure_rotation() {
    // θ=π/2, d=0, a=0, α=0
    let theta = symplex::pi() / symplex::int(2);
    let zero = symplex::int(0);
    let t = dh_matrix(&theta, &zero, &zero, &zero);

    // (0,0) = cos(π/2) = 0
    let r00 = t.get(0, 0).eval().evalf_f64().unwrap();
    assert!(
        r00.abs() < 1e-12,
        "cos(π/2) should be 0, got {r00}"
    );

    // (1,0) = sin(π/2) = 1
    let r10 = t.get(1, 0).eval().evalf_f64().unwrap();
    assert!(
        (r10 - 1.0).abs() < 1e-12,
        "sin(π/2) should be 1, got {r10}"
    );

    // (0,1) = -sin(π/2)cos(0) = -1
    let r01 = t.get(0, 1).eval().evalf_f64().unwrap();
    assert!(
        (r01 + 1.0).abs() < 1e-12,
        "-sin(π/2)cos(0) should be -1, got {r01}"
    );

    // (1,1) = cos(π/2)cos(0) = 0
    let r11 = t.get(1, 1).eval().evalf_f64().unwrap();
    assert!(
        r11.abs() < 1e-12,
        "cos(π/2)cos(0) should be 0, got {r11}"
    );

    // Last row is [0, 0, 0, 1]
    let r33 = t.get(3, 3).eval().evalf_f64().unwrap();
    assert!(
        (r33 - 1.0).abs() < 1e-12,
        "(3,3) should be 1, got {r33}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. DH matrix with translation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dh_matrix_with_translation() {
    // θ=0, d=0, a=1, α=0
    // (0,3) = a·cos(0) = 1
    // (1,3) = a·sin(0) = 0
    let zero = symplex::int(0);
    let one = symplex::int(1);
    let t = dh_matrix(&zero, &zero, &one, &zero);

    let r03 = t.get(0, 3).eval().evalf_f64().unwrap();
    assert!(
        (r03 - 1.0).abs() < 1e-12,
        "a·cos(0) should be 1, got {r03}"
    );

    let r13 = t.get(1, 3).eval().evalf_f64().unwrap();
    assert!(
        r13.abs() < 1e-12,
        "a·sin(0) should be 0, got {r13}"
    );

    // The rotation part should be identity (θ=0, α=0)
    let r00 = t.get(0, 0).eval().evalf_f64().unwrap();
    assert!(
        (r00 - 1.0).abs() < 1e-12,
        "cos(0) should be 1, got {r00}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. FK chain with single joint equals single DH matrix
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fk_chain_single_joint() {
    symplex::vars!(theta1);
    let zero = symplex::int(0);
    let l1 = symplex::var("L1");

    let single_dh = dh_matrix(&theta1, &zero, &l1, &zero);
    let chain = fk_chain(&[(&theta1, &zero, &l1, &zero)]);

    // Substitute concrete values and compare numerically
    let theta_val = symplex::rational(3, 10); // 0.3
    let l_val = symplex::rational(5, 4); // 1.25

    for i in 0..4 {
        for j in 0..4 {
            let a = single_dh
                .get(i, j)
                .subs(&theta1, &theta_val)
                .subs(&l1, &l_val)
                .eval()
                .evalf_f64()
                .unwrap();
            let b = chain
                .get(i, j)
                .subs(&theta1, &theta_val)
                .subs(&l1, &l_val)
                .eval()
                .evalf_f64()
                .unwrap();
            assert!(
                (a - b).abs() < 1e-10,
                "Single joint FK mismatch at ({i},{j}): {a} vs {b}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. FK chain for two-joint planar robot
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fk_chain_two_joint_planar() {
    // Two revolute joints in a plane (α=0 for both).
    // Expected end-effector position:
    //   x = L1·cos(θ1) + L2·cos(θ1+θ2)
    //   y = L1·sin(θ1) + L2·sin(θ1+θ2)
    //   z = 0
    let theta1 = symplex::var("theta1");
    let theta2 = symplex::var("theta2");
    let l1_sym = symplex::var("L1");
    let l2_sym = symplex::var("L2");
    let zero = symplex::int(0);

    let params = [
        (&theta1, &zero, &l1_sym, &zero),
        (&theta2, &zero, &l2_sym, &zero),
    ];
    let t = fk_chain(&params);

    // Test numerically: θ1=0.3, θ2=0.5, L1=1, L2=0.8
    let t1: f64 = 0.3;
    let t2: f64 = 0.5;
    let l1: f64 = 1.0;
    let l2: f64 = 0.8;

    let expected_x = l1 * t1.cos() + l2 * (t1 + t2).cos();
    let expected_y = l1 * t1.sin() + l2 * (t1 + t2).sin();
    let expected_z = 0.0;

    let theta1_val = symplex::rational(3, 10);
    let theta2_val = symplex::rational(1, 2);
    let l1_val = symplex::int(1);
    let l2_val = symplex::rational(4, 5);

    let x_val = t
        .get(0, 3)
        .subs(&theta1, &theta1_val)
        .subs(&theta2, &theta2_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .eval()
        .evalf_f64()
        .unwrap();
    let y_val = t
        .get(1, 3)
        .subs(&theta1, &theta1_val)
        .subs(&theta2, &theta2_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .eval()
        .evalf_f64()
        .unwrap();
    let z_val = t
        .get(2, 3)
        .subs(&theta1, &theta1_val)
        .subs(&theta2, &theta2_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .eval()
        .evalf_f64()
        .unwrap();

    assert!(
        (x_val - expected_x).abs() < 1e-10,
        "x: got {x_val}, expected {expected_x}"
    );
    assert!(
        (y_val - expected_y).abs() < 1e-10,
        "y: got {y_val}, expected {expected_y}"
    );
    assert!(
        z_val.abs() < 1e-10,
        "z: got {z_val}, expected {expected_z}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. fk_position for two-joint planar
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fk_position_two_joint() {
    let theta1 = symplex::var("theta1");
    let theta2 = symplex::var("theta2");
    let l1_sym = symplex::var("L1");
    let l2_sym = symplex::var("L2");
    let zero = symplex::int(0);

    let params = [
        (&theta1, &zero, &l1_sym, &zero),
        (&theta2, &zero, &l2_sym, &zero),
    ];
    let (x, y, z) = fk_position(&params);

    // θ1=0.3, θ2=0.5, L1=1, L2=0.8
    let t1: f64 = 0.3;
    let t2: f64 = 0.5;
    let l1: f64 = 1.0;
    let l2: f64 = 0.8;

    let expected_x = l1 * t1.cos() + l2 * (t1 + t2).cos();
    let expected_y = l1 * t1.sin() + l2 * (t1 + t2).sin();

    let theta1_val = symplex::rational(3, 10);
    let theta2_val = symplex::rational(1, 2);
    let l1_val = symplex::int(1);
    let l2_val = symplex::rational(4, 5);

    let x_val = x
        .subs(&theta1, &theta1_val)
        .subs(&theta2, &theta2_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .eval()
        .evalf_f64()
        .unwrap();
    let y_val = y
        .subs(&theta1, &theta1_val)
        .subs(&theta2, &theta2_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .eval()
        .evalf_f64()
        .unwrap();
    let z_val = z
        .subs(&theta1, &theta1_val)
        .subs(&theta2, &theta2_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .eval()
        .evalf_f64()
        .unwrap();

    assert!(
        (x_val - expected_x).abs() < 1e-10,
        "fk_position x: got {x_val}, expected {expected_x}"
    );
    assert!(
        (y_val - expected_y).abs() < 1e-10,
        "fk_position y: got {y_val}, expected {expected_y}"
    );
    assert!(
        z_val.abs() < 1e-10,
        "fk_position z: got {z_val}, expected 0"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Jacobian of FK position for two-joint planar robot
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fk_jacobian_two_joint() {
    let theta1 = symplex::var("theta1");
    let theta2 = symplex::var("theta2");
    let l1_sym = symplex::var("L1");
    let l2_sym = symplex::var("L2");
    let zero = symplex::int(0);

    let params = [
        (&theta1, &zero, &l1_sym, &zero),
        (&theta2, &zero, &l2_sym, &zero),
    ];
    let (x, y, z) = fk_position(&params);

    // Jacobian of [x, y, z] w.r.t. [θ1, θ2]
    let j = jacobian(&[x, y, z], &[theta1.clone(), theta2.clone()]);

    assert_eq!(j.nrows(), 3);
    assert_eq!(j.ncols(), 2);

    // For a planar robot:
    //   x = L1·cos(θ1) + L2·cos(θ1+θ2)
    //   y = L1·sin(θ1) + L2·sin(θ1+θ2)
    //   z = 0
    //
    // dx/dθ1 = -L1·sin(θ1) - L2·sin(θ1+θ2)
    // dx/dθ2 = -L2·sin(θ1+θ2)
    // dy/dθ1 = L1·cos(θ1) + L2·cos(θ1+θ2)
    // dy/dθ2 = L2·cos(θ1+θ2)
    // dz/dθ1 = 0
    // dz/dθ2 = 0

    let t1: f64 = 0.3;
    let t2: f64 = 0.5;
    let l1: f64 = 1.0;
    let l2: f64 = 0.8;

    let expected_j00 = -l1 * t1.sin() - l2 * (t1 + t2).sin();
    let expected_j01 = -l2 * (t1 + t2).sin();
    let expected_j10 = l1 * t1.cos() + l2 * (t1 + t2).cos();
    let expected_j11 = l2 * (t1 + t2).cos();

    let theta1_val = symplex::rational(3, 10);
    let theta2_val = symplex::rational(1, 2);
    let l1_val = symplex::int(1);
    let l2_val = symplex::rational(4, 5);

    let eval_entry = |i: usize, k: usize| -> f64 {
        j.get(i, k)
            .subs(&theta1, &theta1_val)
            .subs(&theta2, &theta2_val)
            .subs(&l1_sym, &l1_val)
            .subs(&l2_sym, &l2_val)
            .eval()
            .evalf_f64()
            .unwrap()
    };

    let j00 = eval_entry(0, 0);
    let j01 = eval_entry(0, 1);
    let j10 = eval_entry(1, 0);
    let j11 = eval_entry(1, 1);
    let j20 = eval_entry(2, 0);
    let j21 = eval_entry(2, 1);

    assert!(
        (j00 - expected_j00).abs() < 1e-8,
        "J[0,0]: got {j00}, expected {expected_j00}"
    );
    assert!(
        (j01 - expected_j01).abs() < 1e-8,
        "J[0,1]: got {j01}, expected {expected_j01}"
    );
    assert!(
        (j10 - expected_j10).abs() < 1e-8,
        "J[1,0]: got {j10}, expected {expected_j10}"
    );
    assert!(
        (j11 - expected_j11).abs() < 1e-8,
        "J[1,1]: got {j11}, expected {expected_j11}"
    );
    assert!(
        j20.abs() < 1e-10,
        "J[2,0]: got {j20}, expected 0"
    );
    assert!(
        j21.abs() < 1e-10,
        "J[2,1]: got {j21}, expected 0"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. FK chain for three joints
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fk_chain_three_joint() {
    let theta1 = symplex::var("t1");
    let theta2 = symplex::var("t2");
    let theta3 = symplex::var("t3");
    let l1_sym = symplex::var("L1");
    let l2_sym = symplex::var("L2");
    let l3_sym = symplex::var("L3");
    let zero = symplex::int(0);

    let params = [
        (&theta1, &zero, &l1_sym, &zero),
        (&theta2, &zero, &l2_sym, &zero),
        (&theta3, &zero, &l3_sym, &zero),
    ];
    let t = fk_chain(&params);

    assert_eq!(t.shape(), (4, 4));

    // Three-joint planar: x = L1·cos(t1) + L2·cos(t1+t2) + L3·cos(t1+t2+t3)
    //                      y = L1·sin(t1) + L2·sin(t1+t2) + L3·sin(t1+t2+t3)
    let t1v: f64 = 0.2;
    let t2v: f64 = 0.4;
    let t3v: f64 = 0.6;
    let l1: f64 = 1.0;
    let l2: f64 = 0.8;
    let l3: f64 = 0.5;

    let expected_x =
        l1 * t1v.cos() + l2 * (t1v + t2v).cos() + l3 * (t1v + t2v + t3v).cos();
    let expected_y =
        l1 * t1v.sin() + l2 * (t1v + t2v).sin() + l3 * (t1v + t2v + t3v).sin();

    let t1_val = symplex::rational(1, 5);
    let t2_val = symplex::rational(2, 5);
    let t3_val = symplex::rational(3, 5);
    let l1_val = symplex::int(1);
    let l2_val = symplex::rational(4, 5);
    let l3_val = symplex::rational(1, 2);

    let x_val = t
        .get(0, 3)
        .subs(&theta1, &t1_val)
        .subs(&theta2, &t2_val)
        .subs(&theta3, &t3_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .subs(&l3_sym, &l3_val)
        .eval()
        .evalf_f64()
        .unwrap();
    let y_val = t
        .get(1, 3)
        .subs(&theta1, &t1_val)
        .subs(&theta2, &t2_val)
        .subs(&theta3, &t3_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .subs(&l3_sym, &l3_val)
        .eval()
        .evalf_f64()
        .unwrap();

    assert!(
        (x_val - expected_x).abs() < 1e-10,
        "3-joint x: got {x_val}, expected {expected_x}"
    );
    assert!(
        (y_val - expected_y).abs() < 1e-10,
        "3-joint y: got {y_val}, expected {expected_y}"
    );

    // Last row should be [0, 0, 0, 1]
    let r30 = t
        .get(3, 0)
        .subs(&theta1, &t1_val)
        .subs(&theta2, &t2_val)
        .subs(&theta3, &t3_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .subs(&l3_sym, &l3_val)
        .eval()
        .evalf_f64()
        .unwrap();
    let r33 = t
        .get(3, 3)
        .subs(&theta1, &t1_val)
        .subs(&theta2, &t2_val)
        .subs(&theta3, &t3_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .subs(&l3_sym, &l3_val)
        .eval()
        .evalf_f64()
        .unwrap();
    assert!(r30.abs() < 1e-12, "T[3,0] should be 0, got {r30}");
    assert!((r33 - 1.0).abs() < 1e-12, "T[3,3] should be 1, got {r33}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. DH matrix with fully symbolic entries
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dh_matrix_symbolic_entries() {
    let theta = symplex::var("theta");
    let d = symplex::var("d");
    let a = symplex::var("a");
    let alpha = symplex::var("alpha");

    let t = dh_matrix(&theta, &d, &a, &alpha);

    // Shape must be 4×4
    assert_eq!(t.nrows(), 4);
    assert_eq!(t.ncols(), 4);

    // Check that the matrix contains trig functions by looking at string repr
    let s00 = format!("{}", t.get(0, 0));
    assert!(
        s00.contains("cos"),
        "(0,0) should contain cos, got: {s00}"
    );

    let s10 = format!("{}", t.get(1, 0));
    assert!(
        s10.contains("sin"),
        "(1,0) should contain sin, got: {s10}"
    );

    // (3,3) should be 1
    let s33 = format!("{}", t.get(3, 3));
    assert_eq!(s33, "1", "(3,3) should be 1, got: {s33}");

    // (2,0) should be 0
    let s20 = format!("{}", t.get(2, 0));
    assert_eq!(s20, "0", "(2,0) should be 0, got: {s20}");

    // (3,0) should be 0
    let s30 = format!("{}", t.get(3, 0));
    assert_eq!(s30, "0", "(3,0) should be 0, got: {s30}");

    // Verify numerically at a random point
    let tv = symplex::rational(7, 10); // 0.7
    let dv = symplex::rational(3, 10); // 0.3
    let av = symplex::rational(1, 2);  // 0.5
    let alv = symplex::rational(4, 10); // 0.4

    let t_val: f64 = 0.7;
    let d_val: f64 = 0.3;
    let _a_val: f64 = 0.5;
    let _al_val: f64 = 0.4;

    // Check (0,0) = cos(θ)
    let r00 = t
        .get(0, 0)
        .subs(&theta, &tv)
        .subs(&d, &dv)
        .subs(&a, &av)
        .subs(&alpha, &alv)
        .eval()
        .evalf_f64()
        .unwrap();
    assert!(
        (r00 - t_val.cos()).abs() < 1e-10,
        "Symbolic (0,0) = cos(θ): got {r00}, expected {}",
        t_val.cos()
    );

    // Check (2,3) = d
    let r23 = t
        .get(2, 3)
        .subs(&theta, &tv)
        .subs(&d, &dv)
        .subs(&a, &av)
        .subs(&alpha, &alv)
        .eval()
        .evalf_f64()
        .unwrap();
    assert!(
        (r23 - d_val).abs() < 1e-10,
        "Symbolic (2,3) = d: got {r23}, expected {d_val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Rotation submatrix extraction and orthogonality check
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fk_rotation_extraction() {
    let theta = symplex::var("theta");
    let zero = symplex::int(0);
    let l = symplex::var("L");

    let r = fk_rotation(&[(&theta, &zero, &l, &zero)]);

    // Shape must be 3×3
    assert_eq!(r.shape(), (3, 3));

    // Substitute θ=0.6 and verify R^T · R ≈ I numerically
    let theta_val = symplex::rational(3, 5);
    let l_val = symplex::int(1);

    let r_sub = r.subs(&theta, &theta_val).subs(&l, &l_val).eval();
    let r_t = r_sub.transpose();
    let product = r_t.matmul(&r_sub);

    for i in 0..3 {
        for j in 0..3 {
            let val = product.get(i, j).eval().evalf_f64().unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-8,
                "R^T·R[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional: DH matrix with non-zero alpha (link twist)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dh_matrix_with_alpha() {
    // θ=0, d=0, a=0, α=π/2
    // The matrix should be:
    // | 1   0  0  0 |
    // | 0   0 -1  0 |
    // | 0   1  0  0 |
    // | 0   0  0  1 |
    let zero = symplex::int(0);
    let alpha = symplex::pi() / symplex::int(2);
    let t = dh_matrix(&zero, &zero, &zero, &alpha);

    let expected = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    for i in 0..4 {
        for j in 0..4 {
            let val = t.get(i, j).eval().evalf_f64().unwrap();
            assert!(
                (val - expected[i][j]).abs() < 1e-12,
                "α=π/2 entry ({i},{j}): got {val}, expected {}",
                expected[i][j]
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional: DH with d offset along z-axis
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dh_matrix_with_d_offset() {
    // θ=0, d=5, a=0, α=0
    // Should produce identity rotation with d in position (2,3)
    let zero = symplex::int(0);
    let d = symplex::int(5);
    let t = dh_matrix(&zero, &d, &zero, &zero);

    let r23 = t.get(2, 3).eval().evalf_f64().unwrap();
    assert!(
        (r23 - 5.0).abs() < 1e-12,
        "d offset: (2,3) should be 5, got {r23}"
    );

    // Rotation part is still identity
    let r00 = t.get(0, 0).eval().evalf_f64().unwrap();
    assert!(
        (r00 - 1.0).abs() < 1e-12,
        "(0,0) should be 1, got {r00}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional: Two-joint with non-zero alpha (3D robot)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fk_chain_two_joint_3d() {
    // Joint 1: θ1, d=0, a=1, α=π/2
    // Joint 2: θ2, d=0, a=1, α=0
    // This is a simple 2-DOF robot with one out-of-plane twist.
    let theta1 = symplex::var("t1");
    let theta2 = symplex::var("t2");
    let a1 = symplex::int(1);
    let a2 = symplex::int(1);
    let zero = symplex::int(0);
    let alpha1 = symplex::pi() / symplex::int(2);
    let alpha2 = symplex::int(0);

    let params = [
        (&theta1, &zero, &a1, &alpha1),
        (&theta2, &zero, &a2, &alpha2),
    ];
    let t = fk_chain(&params);
    assert_eq!(t.shape(), (4, 4));

    // Numerically check at θ1=0, θ2=0
    // Joint 1 with θ1=0, a=1, α=π/2:
    // T1 = | 1  0  0  1 |
    //      | 0  0 -1  0 |
    //      | 0  1  0  0 |
    //      | 0  0  0  1 |
    //
    // Joint 2 with θ2=0, a=1, α=0:
    // T2 = | 1  -0  0  1 |
    //      | 0   1  0  0 |
    //      | 0   0  1  0 |
    //      | 0   0  0  1 |
    //
    // T = T1 * T2:
    //   (0,3) = T1*(1,0,0,1)^T col3 = 1*1 + 0*0 + 0*0 + 1*1 = 2
    //   (1,3) = 0*1 + 0*0 + (-1)*0 + 0*1 = 0
    //   (2,3) = 0*1 + 1*0 + 0*0 + 0*1 = 0
    let t1_val = symplex::int(0);
    let t2_val = symplex::int(0);

    let x = t
        .get(0, 3)
        .subs(&theta1, &t1_val)
        .subs(&theta2, &t2_val)
        .eval()
        .evalf_f64()
        .unwrap();
    let y = t
        .get(1, 3)
        .subs(&theta1, &t1_val)
        .subs(&theta2, &t2_val)
        .eval()
        .evalf_f64()
        .unwrap();
    let z = t
        .get(2, 3)
        .subs(&theta1, &t1_val)
        .subs(&theta2, &t2_val)
        .eval()
        .evalf_f64()
        .unwrap();

    assert!(
        (x - 2.0).abs() < 1e-10,
        "3D robot x at (0,0): got {x}, expected 2.0"
    );
    assert!(
        y.abs() < 1e-10,
        "3D robot y at (0,0): got {y}, expected 0.0"
    );
    assert!(
        z.abs() < 1e-10,
        "3D robot z at (0,0): got {z}, expected 0.0"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional: rotation orthogonality for multi-joint 3D chain
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fk_rotation_orthogonal_multi_joint() {
    // Two joints with non-zero alpha: rotation should still be orthogonal
    let theta1 = symplex::var("t1");
    let theta2 = symplex::var("t2");
    let zero = symplex::int(0);
    let a1 = symplex::int(1);
    let a2 = symplex::int(1);
    let alpha1 = symplex::pi() / symplex::int(4); // 45 degrees
    let alpha2 = symplex::pi() / symplex::int(3); // 60 degrees

    let params = [
        (&theta1, &zero, &a1, &alpha1),
        (&theta2, &zero, &a2, &alpha2),
    ];
    let r = fk_rotation(&params);
    assert_eq!(r.shape(), (3, 3));

    // Substitute concrete angles and verify R^T · R ≈ I
    let t1_val = symplex::rational(7, 10); // 0.7
    let t2_val = symplex::rational(11, 10); // 1.1

    let r_sub = r.subs(&theta1, &t1_val).subs(&theta2, &t2_val).eval();
    let r_t = r_sub.transpose();
    let product = r_t.matmul(&r_sub);

    for i in 0..3 {
        for j in 0..3 {
            let val = product.get(i, j).eval().evalf_f64().unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-6,
                "Multi-joint R^T·R[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}
