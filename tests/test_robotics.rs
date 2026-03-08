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
            let val = t.get(i, j).eval().eval_f64().unwrap();
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
    let r00 = t.get(0, 0).eval().eval_f64().unwrap();
    assert!(
        r00.abs() < 1e-12,
        "cos(π/2) should be 0, got {r00}"
    );

    // (1,0) = sin(π/2) = 1
    let r10 = t.get(1, 0).eval().eval_f64().unwrap();
    assert!(
        (r10 - 1.0).abs() < 1e-12,
        "sin(π/2) should be 1, got {r10}"
    );

    // (0,1) = -sin(π/2)cos(0) = -1
    let r01 = t.get(0, 1).eval().eval_f64().unwrap();
    assert!(
        (r01 + 1.0).abs() < 1e-12,
        "-sin(π/2)cos(0) should be -1, got {r01}"
    );

    // (1,1) = cos(π/2)cos(0) = 0
    let r11 = t.get(1, 1).eval().eval_f64().unwrap();
    assert!(
        r11.abs() < 1e-12,
        "cos(π/2)cos(0) should be 0, got {r11}"
    );

    // Last row is [0, 0, 0, 1]
    let r33 = t.get(3, 3).eval().eval_f64().unwrap();
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

    let r03 = t.get(0, 3).eval().eval_f64().unwrap();
    assert!(
        (r03 - 1.0).abs() < 1e-12,
        "a·cos(0) should be 1, got {r03}"
    );

    let r13 = t.get(1, 3).eval().eval_f64().unwrap();
    assert!(
        r13.abs() < 1e-12,
        "a·sin(0) should be 0, got {r13}"
    );

    // The rotation part should be identity (θ=0, α=0)
    let r00 = t.get(0, 0).eval().eval_f64().unwrap();
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
                .eval_f64()
                .unwrap();
            let b = chain
                .get(i, j)
                .subs(&theta1, &theta_val)
                .subs(&l1, &l_val)
                .eval()
                .eval_f64()
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
        .eval_f64()
        .unwrap();
    let y_val = t
        .get(1, 3)
        .subs(&theta1, &theta1_val)
        .subs(&theta2, &theta2_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .eval()
        .eval_f64()
        .unwrap();
    let z_val = t
        .get(2, 3)
        .subs(&theta1, &theta1_val)
        .subs(&theta2, &theta2_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .eval()
        .eval_f64()
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
        .eval_f64()
        .unwrap();
    let y_val = y
        .subs(&theta1, &theta1_val)
        .subs(&theta2, &theta2_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .eval()
        .eval_f64()
        .unwrap();
    let z_val = z
        .subs(&theta1, &theta1_val)
        .subs(&theta2, &theta2_val)
        .subs(&l1_sym, &l1_val)
        .subs(&l2_sym, &l2_val)
        .eval()
        .eval_f64()
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
    let j = jacobian(&[&x, &y, &z], &[&theta1, &theta2]);

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
            .eval_f64()
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
        .eval_f64()
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
        .eval_f64()
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
        .eval_f64()
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
        .eval_f64()
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
        .eval_f64()
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
        .eval_f64()
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
    let product = r_t.matmul(&r_sub).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            let val = product.get(i, j).eval().eval_f64().unwrap();
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

    for (i, expected_row) in expected.iter().enumerate() {
        for (j, &exp_val) in expected_row.iter().enumerate() {
            let val = t.get(i, j).eval().eval_f64().unwrap();
            assert!(
                (val - exp_val).abs() < 1e-12,
                "α=π/2 entry ({i},{j}): got {val}, expected {exp_val}",
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

    let r23 = t.get(2, 3).eval().eval_f64().unwrap();
    assert!(
        (r23 - 5.0).abs() < 1e-12,
        "d offset: (2,3) should be 5, got {r23}"
    );

    // Rotation part is still identity
    let r00 = t.get(0, 0).eval().eval_f64().unwrap();
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
        .eval_f64()
        .unwrap();
    let y = t
        .get(1, 3)
        .subs(&theta1, &t1_val)
        .subs(&theta2, &t2_val)
        .eval()
        .eval_f64()
        .unwrap();
    let z = t
        .get(2, 3)
        .subs(&theta1, &t1_val)
        .subs(&theta2, &t2_val)
        .eval()
        .eval_f64()
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
    let product = r_t.matmul(&r_sub).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            let val = product.get(i, j).eval().eval_f64().unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-6,
                "Multi-joint R^T·R[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 1 — Rotation constructors
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rot_x_identity() {
    let zero = symplex::int(0);
    let r = symplex::robotics::rot_x(&zero);
    assert_eq!(r.shape(), (3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let val = r.get(i, j).eval().eval_f64().unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-10,
                "rot_x(0)[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

#[test]
fn rot_x_90deg() {
    // Rx(π/2) = | 1  0   0 |
    //           | 0  0  -1 |
    //           | 0  1   0 |
    let pi = symplex::pi();
    let two = symplex::int(2);
    let angle = &pi / &two;
    let r = symplex::robotics::rot_x(&angle);
    // (1,1) = cos(π/2) = 0
    let val_11 = r.get(1, 1).eval().eval_f64().unwrap();
    assert!(
        val_11.abs() < 1e-10,
        "rot_x(π/2)[1,1] = {val_11}, expected 0"
    );
    // (1,2) = -sin(π/2) = -1
    let val_12 = r.get(1, 2).eval().eval_f64().unwrap();
    assert!(
        (val_12 - (-1.0)).abs() < 1e-10,
        "rot_x(π/2)[1,2] = {val_12}, expected -1"
    );
    // (2,1) = sin(π/2) = 1
    let val_21 = r.get(2, 1).eval().eval_f64().unwrap();
    assert!(
        (val_21 - 1.0).abs() < 1e-10,
        "rot_x(π/2)[2,1] = {val_21}, expected 1"
    );
}

#[test]
fn rot_y_identity() {
    let zero = symplex::int(0);
    let r = symplex::robotics::rot_y(&zero);
    assert_eq!(r.shape(), (3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let val = r.get(i, j).eval().eval_f64().unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-10,
                "rot_y(0)[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

#[test]
fn rot_z_identity() {
    let zero = symplex::int(0);
    let r = symplex::robotics::rot_z(&zero);
    assert_eq!(r.shape(), (3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let val = r.get(i, j).eval().eval_f64().unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-10,
                "rot_z(0)[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

#[test]
fn rot_z_90deg() {
    // Rz(π/2) = | 0  -1  0 |
    //           | 1   0  0 |
    //           | 0   0  1 |
    let pi = symplex::pi();
    let two = symplex::int(2);
    let angle = &pi / &two;
    let r = symplex::robotics::rot_z(&angle);
    // (0,0) = cos(π/2) = 0
    let val_00 = r.get(0, 0).eval().eval_f64().unwrap();
    assert!(
        val_00.abs() < 1e-10,
        "rot_z(π/2)[0,0] = {val_00}, expected 0"
    );
    // (0,1) = -sin(π/2) = -1
    let val_01 = r.get(0, 1).eval().eval_f64().unwrap();
    assert!(
        (val_01 - (-1.0)).abs() < 1e-10,
        "rot_z(π/2)[0,1] = {val_01}, expected -1"
    );
    // (1,0) = sin(π/2) = 1
    let val_10 = r.get(1, 0).eval().eval_f64().unwrap();
    assert!(
        (val_10 - 1.0).abs() < 1e-10,
        "rot_z(π/2)[1,0] = {val_10}, expected 1"
    );
    // (2,2) = 1
    let val_22 = r.get(2, 2).eval().eval_f64().unwrap();
    assert!(
        (val_22 - 1.0).abs() < 1e-10,
        "rot_z(π/2)[2,2] = {val_22}, expected 1"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 1 — skew3
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn skew3_antisymmetric() {
    let a = symplex::var("a");
    let b = symplex::var("b");
    let c = symplex::var("c");
    let s = symplex::robotics::skew3(&a, &b, &c);
    let st = s.transpose();
    let sum = s.add(&st).unwrap();

    // Evaluate at concrete values to verify antisymmetry (S + S^T = 0)
    let a_val = symplex::rational(3, 1);
    let b_val = symplex::rational(5, 1);
    let c_val = symplex::rational(7, 1);
    for i in 0..3 {
        for j in 0..3 {
            let val = sum
                .get(i, j)
                .subs(&a, &a_val)
                .subs(&b, &b_val)
                .subs(&c, &c_val)
                .eval()
                .eval_f64()
                .unwrap();
            assert!(
                val.abs() < 1e-10,
                "skew3 + skew3^T [{i},{j}] = {val}, expected 0"
            );
        }
    }
}

#[test]
fn skew3_cross_product() {
    // skew3(1, 0, 0) * [0, 1, 0]^T should equal [0, 0, 1]^T (i × j = k)
    let one = symplex::int(1);
    let zero = symplex::int(0);
    let s = symplex::robotics::skew3(&one, &zero, &zero);
    let v = symplex::matrix::Matrix::col_vector(vec![zero.clone(), one.clone(), symplex::int(0)]);
    let result = s.matmul(&v).unwrap();
    assert_eq!(result.shape(), (3, 1));
    let r0 = result.get(0, 0).eval().eval_f64().unwrap();
    let r1 = result.get(1, 0).eval().eval_f64().unwrap();
    let r2 = result.get(2, 0).eval().eval_f64().unwrap();
    assert!(
        r0.abs() < 1e-10,
        "cross product x = {r0}, expected 0"
    );
    assert!(
        r1.abs() < 1e-10,
        "cross product y = {r1}, expected 0"
    );
    assert!(
        (r2 - 1.0).abs() < 1e-10,
        "cross product z = {r2}, expected 1"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 1 — homogeneous & translation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn homogeneous_identity() {
    let zero = symplex::int(0);
    let i3 = symplex::matrix::Matrix::identity(3);
    let pos = [zero.clone(), zero.clone(), zero.clone()];
    let h = symplex::robotics::homogeneous(&i3, &pos);
    assert_eq!(h.shape(), (4, 4));
    for i in 0..4 {
        for j in 0..4 {
            let val = h.get(i, j).eval().eval_f64().unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-10,
                "homogeneous(I3, [0,0,0])[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

#[test]
fn homogeneous_translation() {
    let i3 = symplex::matrix::Matrix::identity(3);
    let px = symplex::rational(4, 1);
    let py = symplex::rational(5, 1);
    let pz = symplex::rational(6, 1);
    let pos = [px.clone(), py.clone(), pz.clone()];
    let h = symplex::robotics::homogeneous(&i3, &pos);
    // Last column should be [4, 5, 6, 1]
    let expected_col = [4.0, 5.0, 6.0, 1.0];
    for (i, &exp_val) in expected_col.iter().enumerate() {
        let val = h.get(i, 3).eval().eval_f64().unwrap();
        assert!(
            (val - exp_val).abs() < 1e-10,
            "homogeneous last col [{i}] = {val}, expected {exp_val}",
        );
    }
}

#[test]
fn translation_pure() {
    let t = symplex::robotics::translation(
        &symplex::int(1),
        &symplex::int(2),
        &symplex::int(3),
    );
    assert_eq!(t.shape(), (4, 4));
    // Check last column = [1, 2, 3, 1]
    let expected = [1.0, 2.0, 3.0, 1.0];
    for (i, &exp_val) in expected.iter().enumerate() {
        let val = t.get(i, 3).eval().eval_f64().unwrap();
        assert!(
            (val - exp_val).abs() < 1e-10,
            "translation last col [{i}] = {val}, expected {exp_val}",
        );
    }
    // Check the rotation block is identity
    for i in 0..3 {
        for j in 0..3 {
            let val = t.get(i, j).eval().eval_f64().unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-10,
                "translation rotation[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 1 — Euler angles
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rot_euler_zyx_identity() {
    use symplex::robotics::EulerConvention;
    let zero = symplex::int(0);
    let r = symplex::robotics::rot_euler(&zero, &zero, &zero, EulerConvention::ZYX);
    assert_eq!(r.shape(), (3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let val = r.get(i, j).eval().eval_f64().unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-10,
                "rot_euler(0,0,0,ZYX)[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

#[test]
fn rot_euler_zyx_numerical() {
    use symplex::robotics::EulerConvention;
    // phi=π/2, theta=0, psi=0 → Rz(π/2)·Ry(0)·Rx(0) = Rz(π/2)
    let pi = symplex::pi();
    let two = symplex::int(2);
    let half_pi = &pi / &two;
    let zero = symplex::int(0);

    let r = symplex::robotics::rot_euler(&half_pi, &zero, &zero, EulerConvention::ZYX);
    // Rz(π/2) = | 0  -1  0 |
    //           | 1   0  0 |
    //           | 0   0  1 |
    let expected = [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    for (i, expected_row) in expected.iter().enumerate() {
        for (j, &exp_val) in expected_row.iter().enumerate() {
            let val = r.get(i, j).eval().eval_f64().unwrap();
            assert!(
                (val - exp_val).abs() < 1e-10,
                "rot_euler(π/2,0,0,ZYX)[{i},{j}] = {val}, expected {exp_val}",
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 1 — Matrix powi
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_powi_identity() {
    // M.powi(0) = I for any square matrix
    let a = symplex::var("a");
    let b = symplex::var("b");
    let c = symplex::var("c");
    let d = symplex::var("d");
    let m = symplex::matrix::Matrix::new(vec![
        vec![a.clone(), b.clone()],
        vec![c.clone(), d.clone()],
    ]).unwrap();
    let result = m.powi(0).unwrap();
    assert_eq!(result.shape(), (2, 2));
    for i in 0..2 {
        for j in 0..2 {
            let val = result.get(i, j).eval().eval_f64().unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-10,
                "powi(0)[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

#[test]
fn matrix_powi_one() {
    // M.powi(1) = M (numerically)
    let m = symplex::matrix::Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(3), symplex::int(4)],
    ]).unwrap();
    let result = m.powi(1).unwrap();
    let expected = [[1.0, 2.0], [3.0, 4.0]];
    for (i, expected_row) in expected.iter().enumerate() {
        for (j, &exp_val) in expected_row.iter().enumerate() {
            let val = result.get(i, j).eval().eval_f64().unwrap();
            assert!(
                (val - exp_val).abs() < 1e-10,
                "powi(1)[{i},{j}] = {val}, expected {exp_val}",
            );
        }
    }
}

#[test]
fn matrix_powi_square() {
    // M.powi(2) = M * M
    let m = symplex::matrix::Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(3), symplex::int(4)],
    ]).unwrap();
    let m2 = m.powi(2).unwrap();
    let m_times_m = m.matmul(&m).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let val = m2.get(i, j).eval().eval_f64().unwrap();
            let exp = m_times_m.get(i, j).eval().eval_f64().unwrap();
            assert!(
                (val - exp).abs() < 1e-10,
                "powi(2)[{i},{j}] = {val}, expected {exp}"
            );
        }
    }
}

#[test]
fn matrix_powi_cube() {
    // M.powi(3) = M * M * M (verify numerically)
    // M = | 1 2 |  =>  M^3 = | 37  54 |
    //     | 3 4 |             | 81 118 |
    let m = symplex::matrix::Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(3), symplex::int(4)],
    ]).unwrap();
    let m3 = m.powi(3).unwrap();
    let expected = [[37.0, 54.0], [81.0, 118.0]];
    for (i, expected_row) in expected.iter().enumerate() {
        for (j, &exp_val) in expected_row.iter().enumerate() {
            let val = m3.get(i, j).eval().eval_f64().unwrap();
            assert!(
                (val - exp_val).abs() < 1e-10,
                "powi(3)[{i},{j}] = {val}, expected {exp_val}",
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 1 — diff_with_dependent / eval_derivatives
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_with_dependent_basic() {
    // d/dx(y) with deps={y} should produce Derivative(y, x)
    let x = symplex::var("x");
    let y = symplex::var("y");
    let result = y.diff_with_dependent(&x, &[&y]);
    let s = format!("{result}");
    assert!(
        s.contains("Derivative") || s.contains("d/d"),
        "d/dx(y) with deps={{y}} should be a Derivative node, got: {s}"
    );
}

#[test]
fn diff_with_dependent_implicit() {
    // d/dx(x² + y²) with deps={y} should contain Derivative
    let x = symplex::var("x");
    let y = symplex::var("y");
    let expr = &x.powi(2) + &y.powi(2);
    let result = expr.diff_with_dependent(&x, &[&y]);
    let s = format!("{result}");
    // Should contain both 2*x and a Derivative term involving y
    assert!(
        s.contains("Derivative") || s.contains("d/d"),
        "d/dx(x²+y²) with deps={{y}} should contain Derivative, got: {s}"
    );
}

#[test]
fn eval_derivatives_simple() {
    // eval_derivatives on sin(x).formal_diff(&x) should give cos(x)
    let x = symplex::var("x");
    let expr = x.sin();
    let formal = expr.formal_diff(&x);
    let evald = formal.eval_derivatives();

    // Numerically verify at several points that evald == cos(x)
    let test_vals: &[(i64, i64, f64)] = &[(1, 2, 0.5), (1, 1, 1.0), (2, 1, 2.0), (-1, 1, -1.0)];
    for &(p, q, fval) in test_vals {
        let xv = symplex::rational(p, q);
        let got = evald.subs(&x, &xv).eval().eval_f64().unwrap();
        let expected = fval.cos();
        assert!(
            (got - expected).abs() < 1e-10,
            "eval_derivatives(Derivative(sin(x),x)) at x={fval}: got {got}, expected {expected}"
        );
    }
}
