//! Integration tests for matrix exponential, Kronecker product,
//! and zero-order hold discretization.

mod common;

use symplex::control::StateSpace;
use symplex::matrix::Matrix;

// ═══════════════════════════════════════════════════════════════════════════
// 1. Matrix exponential tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_exp_zero() {
    // exp(0) = I
    let zero = Matrix::zeros(3, 3);
    let result = zero.exp_series(10);
    let ident = Matrix::identity(3);
    for i in 0..3 {
        for j in 0..3 {
            let val = result.get(i, j).evalf_f64().unwrap();
            let expected = ident.get(i, j).evalf_f64().unwrap();
            assert!(
                common::approx_eq(val, expected, 1e-12),
                "exp(0)[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

#[test]
fn matrix_exp_identity_scaled() {
    // exp(t·I) ≈ eᵗ·I for t = 0.1
    let t_val = 0.1_f64;
    let t = symplex::rational(1, 10); // exact 1/10
    let n = 3;
    let ti = Matrix::identity(n).scale(&t);
    let result = ti.exp_series(15);

    let expected_diag = t_val.exp(); // e^0.1 ≈ 1.10517...
    for i in 0..n {
        for j in 0..n {
            let val = result.get(i, j).eval().evalf_f64().unwrap();
            if i == j {
                assert!(
                    common::approx_eq(val, expected_diag, 1e-10),
                    "exp(t·I)[{i},{i}] = {val}, expected {expected_diag}"
                );
            } else {
                assert!(
                    common::approx_eq(val, 0.0, 1e-10),
                    "exp(t·I)[{i},{j}] = {val}, expected 0"
                );
            }
        }
    }
}

#[test]
fn matrix_exp_nilpotent() {
    // N = [[0,1],[0,0]], N² = 0, so exp(N) = I + N (exact for order ≥ 2)
    let n = Matrix::new(vec![
        vec![symplex::int(0), symplex::int(1)],
        vec![symplex::int(0), symplex::int(0)],
    ]);
    let result = n.exp_series(2);

    // Expected: [[1, 1], [0, 1]]
    let expected = [
        [1.0, 1.0],
        [0.0, 1.0],
    ];
    for i in 0..2 {
        for j in 0..2 {
            let val = result.get(i, j).eval().evalf_f64().unwrap();
            assert!(
                common::approx_eq(val, expected[i][j], 1e-14),
                "exp(N)[{i},{j}] = {val}, expected {}",
                expected[i][j]
            );
        }
    }
}

#[test]
fn matrix_exp_diagonal() {
    // exp(diag(a, b)) = diag(eᵃ, eᵇ)
    // Use a = 1/2, b = -1/3
    let a_val = 0.5_f64;
    let b_val: f64 = -1.0 / 3.0;
    let m = Matrix::new(vec![
        vec![symplex::rational(1, 2), symplex::int(0)],
        vec![symplex::int(0), symplex::rational(-1, 3)],
    ]);
    let result = m.exp_series(15);

    let expected_00 = a_val.exp();
    let expected_11 = b_val.exp();
    let val_00 = result.get(0, 0).eval().evalf_f64().unwrap();
    let val_01 = result.get(0, 1).eval().evalf_f64().unwrap();
    let val_10 = result.get(1, 0).eval().evalf_f64().unwrap();
    let val_11 = result.get(1, 1).eval().evalf_f64().unwrap();

    assert!(
        common::approx_eq(val_00, expected_00, 1e-8),
        "exp(diag)[0,0] = {val_00}, expected {expected_00}"
    );
    assert!(
        common::approx_eq(val_11, expected_11, 1e-8),
        "exp(diag)[1,1] = {val_11}, expected {expected_11}"
    );
    assert!(
        common::approx_eq(val_01, 0.0, 1e-10),
        "exp(diag)[0,1] = {val_01}, expected 0"
    );
    assert!(
        common::approx_eq(val_10, 0.0, 1e-10),
        "exp(diag)[1,0] = {val_10}, expected 0"
    );
}

#[test]
fn matrix_exp_2x2_numerical() {
    // exp([[0,1],[0,0]]) = [[1,1],[0,1]] (nilpotent, exact)
    let m = Matrix::new(vec![
        vec![symplex::int(0), symplex::int(1)],
        vec![symplex::int(0), symplex::int(0)],
    ]);
    let result = m.exp_series(10);

    let val_00 = result.get(0, 0).eval().evalf_f64().unwrap();
    let val_01 = result.get(0, 1).eval().evalf_f64().unwrap();
    let val_10 = result.get(1, 0).eval().evalf_f64().unwrap();
    let val_11 = result.get(1, 1).eval().evalf_f64().unwrap();

    assert!(common::approx_eq(val_00, 1.0, 1e-14), "got {val_00}");
    assert!(common::approx_eq(val_01, 1.0, 1e-14), "got {val_01}");
    assert!(common::approx_eq(val_10, 0.0, 1e-14), "got {val_10}");
    assert!(common::approx_eq(val_11, 1.0, 1e-14), "got {val_11}");
}

#[test]
fn matrix_exp_series_converges() {
    // Compare order 5 vs order 15 on a specific matrix.
    // Higher order should be closer to the true value.
    // Use M = [[0.1, 0.2],[0.3, 0.1]]
    let m = Matrix::new(vec![
        vec![symplex::rational(1, 10), symplex::rational(1, 5)],
        vec![symplex::rational(3, 10), symplex::rational(1, 10)],
    ]);

    let low = m.exp_series(5);
    let high = m.exp_series(15);

    // The true exp(M) can be computed; for a small matrix these should agree
    // closely but not identically at order 5. At order 15 it should be very precise.
    // Just check that the results are numerically close (converging).
    for i in 0..2 {
        for j in 0..2 {
            let v_low = low.get(i, j).eval().evalf_f64().unwrap();
            let v_high = high.get(i, j).eval().evalf_f64().unwrap();
            // They should be close (within ~1e-4 for this small matrix)
            let diff = (v_low - v_high).abs();
            assert!(
                diff < 1e-4,
                "exp_series convergence: [{i},{j}] order5={v_low}, order15={v_high}, diff={diff}"
            );
            // The high order should be at least as close to itself (sanity)
            assert!(v_high.is_finite(), "high-order result should be finite");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Kronecker product tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn kronecker_dimensions() {
    // (2×3) ⊗ (4×5) → (8×15)
    let a = Matrix::from_fn(2, 3, |_i, _j| symplex::int(1));
    let b = Matrix::from_fn(4, 5, |_i, _j| symplex::int(1));
    let c = a.kronecker(&b);
    assert_eq!(c.nrows(), 8);
    assert_eq!(c.ncols(), 15);
}

#[test]
fn kronecker_identity() {
    // A ⊗ I₂ should produce a block-diagonal-like structure
    // For A = [[a, b],[c, d]], A ⊗ I₂ = [[a·I, b·I],[c·I, d·I]]
    let a = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(3), symplex::int(4)],
    ]);
    let i2 = Matrix::identity(2);
    let result = a.kronecker(&i2);

    assert_eq!(result.nrows(), 4);
    assert_eq!(result.ncols(), 4);

    // result[0,0] = 1*1 = 1, result[0,1] = 1*0 = 0, result[0,2] = 2*1 = 2, result[0,3] = 2*0 = 0
    let expected = [
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 1.0, 0.0, 2.0],
        [3.0, 0.0, 4.0, 0.0],
        [0.0, 3.0, 0.0, 4.0],
    ];
    for i in 0..4 {
        for j in 0..4 {
            let val = result.get(i, j).eval().evalf_f64().unwrap();
            assert!(
                common::approx_eq(val, expected[i][j], 1e-14),
                "A⊗I[{i},{j}] = {val}, expected {}",
                expected[i][j]
            );
        }
    }
}

#[test]
fn kronecker_scalar() {
    // (1×1 scalar s) ⊗ B = B scaled by s
    let s = Matrix::new(vec![vec![symplex::int(3)]]);
    let b = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(4), symplex::int(5)],
    ]);
    let result = s.kronecker(&b);

    assert_eq!(result.nrows(), 2);
    assert_eq!(result.ncols(), 2);

    let expected = [[3.0, 6.0], [12.0, 15.0]];
    for i in 0..2 {
        for j in 0..2 {
            let val = result.get(i, j).eval().evalf_f64().unwrap();
            assert!(
                common::approx_eq(val, expected[i][j], 1e-14),
                "scalar⊗B[{i},{j}] = {val}, expected {}",
                expected[i][j]
            );
        }
    }
}

#[test]
fn kronecker_known_values() {
    // [[1,2],[3,4]] ⊗ [[0,5],[6,7]]
    // = [[1*[[0,5],[6,7]], 2*[[0,5],[6,7]]],
    //    [3*[[0,5],[6,7]], 4*[[0,5],[6,7]]]]
    // = [[ 0,  5,  0, 10],
    //    [ 6,  7, 12, 14],
    //    [ 0, 15,  0, 20],
    //    [18, 21, 24, 28]]
    let a = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(3), symplex::int(4)],
    ]);
    let b = Matrix::new(vec![
        vec![symplex::int(0), symplex::int(5)],
        vec![symplex::int(6), symplex::int(7)],
    ]);
    let result = a.kronecker(&b);

    assert_eq!(result.nrows(), 4);
    assert_eq!(result.ncols(), 4);

    let expected = [
        [0.0, 5.0, 0.0, 10.0],
        [6.0, 7.0, 12.0, 14.0],
        [0.0, 15.0, 0.0, 20.0],
        [18.0, 21.0, 24.0, 28.0],
    ];
    for i in 0..4 {
        for j in 0..4 {
            let val = result.get(i, j).eval().evalf_f64().unwrap();
            assert!(
                common::approx_eq(val, expected[i][j], 1e-14),
                "kronecker[{i},{j}] = {val}, expected {}",
                expected[i][j]
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Discretization tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn discretize_zoh_simple() {
    // Discretize a simple 2-state, 1-input system and check dimensions.
    let a = Matrix::new(vec![
        vec![symplex::int(0), symplex::int(1)],
        vec![symplex::int(-2), symplex::int(-3)],
    ]);
    let b = Matrix::new(vec![
        vec![symplex::int(0)],
        vec![symplex::int(1)],
    ]);
    let c = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(0)],
    ]);
    let d = Matrix::new(vec![vec![symplex::int(0)]]);
    let ss = StateSpace::new(a, b, c, d);

    let dt = symplex::rational(1, 10); // dt = 0.1
    let ss_d = ss.discretize_zoh(&dt, 10);

    assert_eq!(ss_d.num_states(), 2);
    assert_eq!(ss_d.num_inputs(), 1);
    assert_eq!(ss_d.num_outputs(), 1);
    assert_eq!(ss_d.a.nrows(), 2);
    assert_eq!(ss_d.a.ncols(), 2);
    assert_eq!(ss_d.b.nrows(), 2);
    assert_eq!(ss_d.b.ncols(), 1);
}

#[test]
fn discretize_zoh_integrator() {
    // Double integrator: A = [[0,1],[0,0]], B = [[0],[1]]
    // This is nilpotent (A² = 0), so:
    //   eᴬᵈᵗ = I + A·dt = [[1, dt], [0, 1]]
    //   Bᵈ = (I·dt + A·dt²/2)·B = [[dt²/2], [dt]]
    let a = Matrix::new(vec![
        vec![symplex::int(0), symplex::int(1)],
        vec![symplex::int(0), symplex::int(0)],
    ]);
    let b = Matrix::new(vec![
        vec![symplex::int(0)],
        vec![symplex::int(1)],
    ]);
    let c = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(0)],
    ]);
    let d = Matrix::new(vec![vec![symplex::int(0)]]);
    let ss = StateSpace::new(a, b, c, d);

    let dt = symplex::rational(1, 10); // dt = 0.1
    let ss_d = ss.discretize_zoh(&dt, 10);

    let dt_val = 0.1_f64;

    // Check Aᵈ = [[1, 0.1], [0, 1]]
    let ad_00 = ss_d.a.get(0, 0).eval().evalf_f64().unwrap();
    let ad_01 = ss_d.a.get(0, 1).eval().evalf_f64().unwrap();
    let ad_10 = ss_d.a.get(1, 0).eval().evalf_f64().unwrap();
    let ad_11 = ss_d.a.get(1, 1).eval().evalf_f64().unwrap();

    assert!(
        common::approx_eq(ad_00, 1.0, 1e-10),
        "Ad[0,0] = {ad_00}, expected 1.0"
    );
    assert!(
        common::approx_eq(ad_01, dt_val, 1e-10),
        "Ad[0,1] = {ad_01}, expected {dt_val}"
    );
    assert!(
        common::approx_eq(ad_10, 0.0, 1e-10),
        "Ad[1,0] = {ad_10}, expected 0.0"
    );
    assert!(
        common::approx_eq(ad_11, 1.0, 1e-10),
        "Ad[1,1] = {ad_11}, expected 1.0"
    );

    // Check Bᵈ = [[dt²/2], [dt]] = [[0.005], [0.1]]
    let bd_00 = ss_d.b.get(0, 0).eval().evalf_f64().unwrap();
    let bd_10 = ss_d.b.get(1, 0).eval().evalf_f64().unwrap();

    assert!(
        common::approx_eq(bd_00, dt_val * dt_val / 2.0, 1e-10),
        "Bd[0,0] = {bd_00}, expected {}",
        dt_val * dt_val / 2.0
    );
    assert!(
        common::approx_eq(bd_10, dt_val, 1e-10),
        "Bd[1,0] = {bd_10}, expected {dt_val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Additional edge-case tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_exp_1x1() {
    // exp([[a]]) = [[eᵃ]] — check with a = 1/2
    let m = Matrix::new(vec![vec![symplex::rational(1, 2)]]);
    let result = m.exp_series(15);
    let val = result.get(0, 0).eval().evalf_f64().unwrap();
    let expected = 0.5_f64.exp();
    assert!(
        common::approx_eq(val, expected, 1e-10),
        "exp([[1/2]])[0,0] = {val}, expected {expected}"
    );
}

#[test]
fn kronecker_1x1_times_1x1() {
    // (1×1) ⊗ (1×1) = (1×1) with product of entries
    let a = Matrix::new(vec![vec![symplex::int(3)]]);
    let b = Matrix::new(vec![vec![symplex::int(7)]]);
    let result = a.kronecker(&b);
    assert_eq!(result.nrows(), 1);
    assert_eq!(result.ncols(), 1);
    let val = result.get(0, 0).eval().evalf_f64().unwrap();
    assert!(
        common::approx_eq(val, 21.0, 1e-14),
        "3⊗7 = {val}, expected 21"
    );
}

#[test]
fn matrix_exp_negative_entries() {
    // exp(diag(-1, -2)) = diag(e⁻¹, e⁻²)
    let m = Matrix::new(vec![
        vec![symplex::int(-1), symplex::int(0)],
        vec![symplex::int(0), symplex::int(-2)],
    ]);
    let result = m.exp_series(20);
    let val_00 = result.get(0, 0).eval().evalf_f64().unwrap();
    let val_11 = result.get(1, 1).eval().evalf_f64().unwrap();
    let val_01 = result.get(0, 1).eval().evalf_f64().unwrap();
    let val_10 = result.get(1, 0).eval().evalf_f64().unwrap();

    assert!(
        common::approx_eq(val_00, (-1.0_f64).exp(), 1e-8),
        "exp(-1) = {val_00}, expected {}",
        (-1.0_f64).exp()
    );
    assert!(
        common::approx_eq(val_11, (-2.0_f64).exp(), 1e-6),
        "exp(-2) = {val_11}, expected {}",
        (-2.0_f64).exp()
    );
    assert!(common::approx_eq(val_01, 0.0, 1e-10));
    assert!(common::approx_eq(val_10, 0.0, 1e-10));
}
