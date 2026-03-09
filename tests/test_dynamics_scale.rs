//! Dynamics scaling experiments: can symplex handle real-world robot arms?
//!
//! Experiment 1: 3-DOF planar arm — full Lagrangian dynamics
//! Experiment 2: 6-DOF PUMA-like spatial arm — FK chain, Jacobian, codegen

use std::time::Instant;

use symplex::dynamics::{euler_lagrange, mass_matrix, manipulator_equation};
use symplex::matrix::{jacobian, Matrix};
use symplex::prelude::*;
use symplex::robotics::{dh_matrix, fk_chain, fk_position};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn assert_near(a: f64, b: f64, tol: f64, msg: &str) {
    assert!(
        (a - b).abs() < tol,
        "{msg}: got {a}, expected {b}, diff={}",
        (a - b).abs()
    );
}

/// Count total ops across all entries of a matrix.
fn matrix_total_ops(m: &Matrix) -> usize {
    let (nr, nc) = m.shape();
    let mut total = 0;
    for i in 0..nr {
        for j in 0..nc {
            total += m.get(i, j).count_ops();
        }
    }
    total
}

/// Count total term_count across all entries of a matrix.
fn matrix_total_terms(m: &Matrix) -> usize {
    let (nr, nc) = m.shape();
    let mut total = 0;
    for i in 0..nr {
        for j in 0..nc {
            total += m.get(i, j).term_count();
        }
    }
    total
}

// ═══════════════════════════════════════════════════════════════════════════
// Experiment 1: 3-DOF Planar Arm
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn experiment_3dof_planar_arm_dynamics() {
    let ctx = Context::new();
    println!("\n{}", "=".repeat(60));
    println!("  EXPERIMENT 1: 3-DOF Planar Arm Dynamics");
    println!("{}\n", "=".repeat(60));

    // ── Symbols ──────────────────────────────────────────────────────────
    let m1 = ctx.symbol("m1");
    let m2 = ctx.symbol("m2");
    let m3 = ctx.symbol("m3");
    let l1 = ctx.symbol("L1");
    let l2 = ctx.symbol("L2");
    let l3 = ctx.symbol("L3");
    let g_sym = ctx.symbol("g");

    let q1 = ctx.symbol("q1");
    let q2 = ctx.symbol("q2");
    let q3 = ctx.symbol("q3");
    let qd1 = ctx.symbol("qd1");
    let qd2 = ctx.symbol("qd2");
    let qd3 = ctx.symbol("qd3");
    let qdd1 = ctx.symbol("qdd1");
    let qdd2 = ctx.symbol("qdd2");
    let qdd3 = ctx.symbol("qdd3");

    let half = ctx.rational(1, 2);

    // ── Step 1: Forward kinematics — COM positions ───────────────────────
    println!("Step 1: Building COM positions for 3 links...");
    let t0 = Instant::now();

    // Link 1 COM: midpoint of link 1
    //   x1 = (L1/2) cos(q1)
    //   y1 = (L1/2) sin(q1)
    let x1 = &(&half * &l1) * &q1.cos();
    let y1 = &(&half * &l1) * &q1.sin();

    // Link 2 COM: end of link 1 + midpoint of link 2
    //   q12 = q1 + q2
    //   x2 = L1 cos(q1) + (L2/2) cos(q1+q2)
    //   y2 = L1 sin(q1) + (L2/2) sin(q1+q2)
    let q12 = &q1 + &q2;
    let x2 = &(&l1 * &q1.cos()) + &(&(&half * &l2) * &q12.cos());
    let y2 = &(&l1 * &q1.sin()) + &(&(&half * &l2) * &q12.sin());

    // Link 3 COM: end of link 1 + end of link 2 + midpoint of link 3
    //   q123 = q1 + q2 + q3
    //   x3 = L1 cos(q1) + L2 cos(q1+q2) + (L3/2) cos(q1+q2+q3)
    //   y3 = L1 sin(q1) + L2 sin(q1+q2) + (L3/2) sin(q1+q2+q3)
    let q123 = &(&q1 + &q2) + &q3;
    let x3 = &(&l1 * &q1.cos()) + &(&(&l2 * &q12.cos()) + &(&(&half * &l3) * &q123.cos()));
    let y3 = &(&l1 * &q1.sin()) + &(&(&l2 * &q12.sin()) + &(&(&half * &l3) * &q123.sin()));

    let fk_time = t0.elapsed();
    println!("  FK positions built in {:?}", fk_time);
    println!(
        "  x3 ops={}, terms={}",
        x3.count_ops(),
        x3.term_count()
    );
    println!(
        "  y3 ops={}, terms={}",
        y3.count_ops(),
        y3.term_count()
    );

    // ── Step 2: COM velocities ───────────────────────────────────────────
    //
    // ẋ_i = Σ_j (∂x_i/∂q_j) · q̇_j
    // We compute these using the chain rule manually.
    println!("\nStep 2: Building COM velocities...");
    let t0 = Instant::now();

    // Link 1 velocity
    //   ẋ1 = ∂x1/∂q1 · qd1  = -(L1/2) sin(q1) · qd1
    //   ẏ1 = ∂y1/∂q1 · qd1  =  (L1/2) cos(q1) · qd1
    let xd1 = &x1.diff(&q1) * &qd1;
    let yd1 = &y1.diff(&q1) * &qd1;

    // Link 2 velocity
    //   ẋ2 = ∂x2/∂q1 · qd1 + ∂x2/∂q2 · qd2
    //   ẏ2 = ∂y2/∂q1 · qd1 + ∂y2/∂q2 · qd2
    let xd2 = &(&x2.diff(&q1) * &qd1) + &(&x2.diff(&q2) * &qd2);
    let yd2 = &(&y2.diff(&q1) * &qd1) + &(&y2.diff(&q2) * &qd2);

    // Link 3 velocity
    //   ẋ3 = ∂x3/∂q1 · qd1 + ∂x3/∂q2 · qd2 + ∂x3/∂q3 · qd3
    //   ẏ3 = ∂y3/∂q1 · qd1 + ∂y3/∂q2 · qd2 + ∂y3/∂q3 · qd3
    let xd3 = &(&(&x3.diff(&q1) * &qd1) + &(&x3.diff(&q2) * &qd2))
        + &(&x3.diff(&q3) * &qd3);
    let yd3 = &(&(&y3.diff(&q1) * &qd1) + &(&y3.diff(&q2) * &qd2))
        + &(&y3.diff(&q3) * &qd3);

    let vel_time = t0.elapsed();
    println!("  Velocities built in {:?}", vel_time);
    println!(
        "  xd3 ops={}, terms={}",
        xd3.count_ops(),
        xd3.term_count()
    );

    // ── Step 3: Kinetic energy T = ½ Σ m_i (ẋ_i² + ẏ_i²) ──────────────
    println!("\nStep 3: Building kinetic energy T...");
    let t0 = Instant::now();

    let v1_sq = &(&xd1 * &xd1) + &(&yd1 * &yd1);
    let v2_sq = &(&xd2 * &xd2) + &(&yd2 * &yd2);
    let v3_sq = &(&xd3 * &xd3) + &(&yd3 * &yd3);

    let ke = &(&(&half * &m1) * &v1_sq)
        + &(&(&(&half * &m2) * &v2_sq) + &(&(&half * &m3) * &v3_sq));

    let ke_build_time = t0.elapsed();
    println!("  T built in {:?}", ke_build_time);
    println!("  T ops={}, terms={}", ke.count_ops(), ke.term_count());

    // ── Step 3b: Expand T so derivatives work on polynomial-like form ────
    println!("\nStep 3b: Expanding T...");
    let t0 = Instant::now();
    let ke_expanded = ke.expand();
    let ke_expand_time = t0.elapsed();
    println!("  T expanded in {:?}", ke_expand_time);
    println!(
        "  T_expanded ops={}, terms={}",
        ke_expanded.count_ops(),
        ke_expanded.term_count()
    );

    // ── Step 4: Potential energy V = Σ m_i g y_i ────────────────────────
    println!("\nStep 4: Building potential energy V...");
    let t0 = Instant::now();

    let pe = &(&(&m1 * &g_sym) * &y1)
        + &(&(&(&m2 * &g_sym) * &y2) + &(&(&m3 * &g_sym) * &y3));

    let pe_build_time = t0.elapsed();
    println!("  V built in {:?}", pe_build_time);
    println!("  V ops={}, terms={}", pe.count_ops(), pe.term_count());

    // ── Step 5: Mass matrix M(q) ────────────────────────────────────────
    println!("\nStep 5: Computing mass matrix M(q) via ∂²T/∂q̇ᵢ∂q̇ⱼ...");
    let t0 = Instant::now();

    let mm = mass_matrix(&ke_expanded, &[&qd1, &qd2, &qd3]);

    let mm_time = t0.elapsed();
    println!("  Mass matrix computed in {:?}", mm_time);
    assert_eq!(mm.shape(), (3, 3));
    println!(
        "  M total ops={}, total terms={}",
        matrix_total_ops(&mm),
        matrix_total_terms(&mm)
    );
    for i in 0..3 {
        for j in 0..3 {
            println!(
                "    M[{i},{j}]: ops={}, terms={}",
                mm.get(i, j).count_ops(),
                mm.get(i, j).term_count()
            );
        }
    }

    // ── Step 6: Euler-Lagrange equations ────────────────────────────────
    println!("\nStep 6: Computing Euler-Lagrange equations...");
    let t0 = Instant::now();

    let coords: [(&Ex, &Ex); 3] = [(&q1, &qd1), (&q2, &qd2), (&q3, &qd3)];
    let accels: [&Ex; 3] = [&qdd1, &qdd2, &qdd3];
    let eqs = euler_lagrange(&ke_expanded, &pe, &coords, &accels);

    let el_time = t0.elapsed();
    println!("  Euler-Lagrange computed in {:?}", el_time);
    assert_eq!(eqs.len(), 3);
    for (i, eq) in eqs.iter().enumerate() {
        println!(
            "    EL[{i}]: ops={}, terms={}",
            eq.count_ops(),
            eq.term_count()
        );
    }

    // ── Step 7: Simplify one mass matrix entry with trigsimp ────────────
    println!("\nStep 7: Simplifying M[0,0] with trigsimp...");
    let t0 = Instant::now();
    let m00_simplified = mm.get(0, 0).simplify_trig();
    let trigsimp_time = t0.elapsed();
    println!("  trigsimp(M[0,0]) completed in {:?}", trigsimp_time);
    println!(
        "  Before: ops={}, terms={}",
        mm.get(0, 0).count_ops(),
        mm.get(0, 0).term_count()
    );
    println!(
        "  After:  ops={}, terms={}",
        m00_simplified.count_ops(),
        m00_simplified.term_count()
    );

    // ── Step 8: Spot-check numerical correctness ────────────────────────
    println!("\nStep 8: Numerical correctness check...");

    // Use specific numeric values
    let vals: &[(&Ex, f64)] = &[
        (&m1, 2.0),
        (&m2, 1.5),
        (&m3, 1.0),
        (&l1, 1.0),
        (&l2, 0.8),
        (&l3, 0.5),
        (&g_sym, 9.81),
        (&q1, 0.3),
        (&q2, 0.5),
        (&q3, -0.2),
        (&qd1, 1.0),
        (&qd2, -0.5),
        (&qd3, 0.3),
        (&qdd1, 0.0),
        (&qdd2, 0.0),
        (&qdd3, 0.0),
    ];

    let subs_numeric = |e: &Ex| -> f64 {
        let mut result = e.clone();
        for &(var, val) in vals {
            let num = if val == val.floor() && val.abs() < 1e9 {
                ctx.int(val as i64)
            } else {
                // Use rational approximation for decimals
                ctx.rational((val * 10000.0) as i64, 10000)
            };
            result = result.subs(var, &num);
        }
        result.eval().eval_f64().unwrap()
    };

    // Check M is symmetric: M[0,1] == M[1,0]
    let m01_val = subs_numeric(mm.get(0, 1));
    let m10_val = subs_numeric(mm.get(1, 0));
    assert_near(m01_val, m10_val, 1e-8, "M must be symmetric: M[0,1] vs M[1,0]");

    let m02_val = subs_numeric(mm.get(0, 2));
    let m20_val = subs_numeric(mm.get(2, 0));
    assert_near(m02_val, m20_val, 1e-8, "M must be symmetric: M[0,2] vs M[2,0]");

    let m12_val = subs_numeric(mm.get(1, 2));
    let m21_val = subs_numeric(mm.get(2, 1));
    assert_near(m12_val, m21_val, 1e-8, "M must be symmetric: M[1,2] vs M[2,1]");
    println!("  ✓ Mass matrix is symmetric");

    // Check M[0,0] > 0 (positive definite diagonal)
    let m00_val = subs_numeric(mm.get(0, 0));
    assert!(m00_val > 0.0, "M[0,0] must be positive, got {m00_val}");
    let m11_val = subs_numeric(mm.get(1, 1));
    assert!(m11_val > 0.0, "M[1,1] must be positive, got {m11_val}");
    let m22_val = subs_numeric(mm.get(2, 2));
    assert!(m22_val > 0.0, "M[2,2] must be positive, got {m22_val}");
    println!("  ✓ Diagonal entries are positive: M[0,0]={m00_val:.6}, M[1,1]={m11_val:.6}, M[2,2]={m22_val:.6}");

    // Verify M[0,0] against hand-computed value
    // M[0,0] = m1*(L1/2)^2 + m2*(L1^2 + (L2/2)^2 + 2*L1*(L2/2)*cos(q2))
    //        + m3*(L1^2 + L2^2 + (L3/2)^2 + 2*L1*L2*cos(q2) + 2*L1*(L3/2)*cos(q2+q3) + 2*L2*(L3/2)*cos(q3))
    let (mv1, mv2, mv3) = (2.0_f64, 1.5_f64, 1.0_f64);
    let (lv1, lv2, lv3) = (1.0_f64, 0.8_f64, 0.5_f64);
    let (_qv1, qv2, qv3) = (0.3_f64, 0.5_f64, -0.2_f64);
    let expected_m00 = mv1 * (lv1 / 2.0).powi(2)
        + mv2 * (lv1.powi(2) + (lv2 / 2.0).powi(2) + 2.0 * lv1 * (lv2 / 2.0) * qv2.cos())
        + mv3
            * (lv1.powi(2)
                + lv2.powi(2)
                + (lv3 / 2.0).powi(2)
                + 2.0 * lv1 * lv2 * qv2.cos()
                + 2.0 * lv1 * (lv3 / 2.0) * (qv2 + qv3).cos()
                + 2.0 * lv2 * (lv3 / 2.0) * qv3.cos());
    assert_near(
        m00_val,
        expected_m00,
        1e-6,
        "M[0,0] vs hand-computed value",
    );
    println!("  ✓ M[0,0] = {m00_val:.6} matches expected {expected_m00:.6}");

    // ── Step 9: Full manipulator equation ────────────────────────────────
    println!("\nStep 9: Computing full manipulator equation M, C, g...");
    let t0 = Instant::now();
    let (mm2, coriolis, grav) = manipulator_equation(
        &ke_expanded,
        &pe,
        &[&q1, &q2, &q3],
        &[&qd1, &qd2, &qd3],
    );
    let manip_time = t0.elapsed();
    println!("  manipulator_equation() completed in {:?}", manip_time);
    assert_eq!(mm2.shape(), (3, 3));
    assert_eq!(coriolis.shape(), (3, 3));
    assert_eq!(grav.len(), 3);
    println!(
        "  M total ops={}, C total ops={}, g total ops={}",
        matrix_total_ops(&mm2),
        matrix_total_ops(&coriolis),
        grav.iter().map(|e| e.count_ops()).sum::<usize>()
    );

    // ── Summary ──────────────────────────────────────────────────────────
    println!("\n{}", "─".repeat(60));
    println!("  3-DOF TIMING SUMMARY");
    println!("{}", "─".repeat(60));
    println!("  FK positions:         {:>10?}", fk_time);
    println!("  COM velocities:       {:>10?}", vel_time);
    println!("  Build T:              {:>10?}", ke_build_time);
    println!("  Expand T:             {:>10?}", ke_expand_time);
    println!("  Build V:              {:>10?}", pe_build_time);
    println!("  mass_matrix():        {:>10?}", mm_time);
    println!("  euler_lagrange():     {:>10?}", el_time);
    println!("  trigsimp(M[0,0]):     {:>10?}", trigsimp_time);
    println!("  manipulator_equation:{:>10?}", manip_time);
    let total_3dof = fk_time
        + vel_time
        + ke_build_time
        + ke_expand_time
        + pe_build_time
        + mm_time
        + el_time
        + trigsimp_time
        + manip_time;
    println!("  ──────────────────────────────");
    println!("  TOTAL:                {:>10?}", total_3dof);
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// Experiment 2: 6-DOF PUMA-like Spatial Arm
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn experiment_6dof_puma_fk_jacobian_codegen() {
    let ctx = Context::new();
    println!("\n{}", "=".repeat(60));
    println!("  EXPERIMENT 2: 6-DOF PUMA-like FK / Jacobian / Codegen");
    println!("{}\n", "=".repeat(60));

    // ── Symbols ──────────────────────────────────────────────────────────
    let q1 = ctx.symbol("q1");
    let q2 = ctx.symbol("q2");
    let q3 = ctx.symbol("q3");
    let q4 = ctx.symbol("q4");
    let q5 = ctx.symbol("q5");
    let q6 = ctx.symbol("q6");

    let a2 = ctx.symbol("a2");
    let d4 = ctx.symbol("d4");

    let zero = ctx.int(0);
    let pi = ctx.pi();
    let half_pi = &ctx.rational(1, 2) * &pi;
    let neg_half_pi = &ctx.rational(-1, 2) * &ctx.pi();

    // ── Step 1: Build individual DH matrices ─────────────────────────────
    // PUMA-like DH parameters:
    // | Joint | θ  | d  | a  | α     |
    // |-------|----|----|----| ------|
    // | 1     | q1 | 0  | 0  | π/2   |
    // | 2     | q2 | 0  | a2 | 0     |
    // | 3     | q3 | 0  | 0  | π/2   |
    // | 4     | q4 | d4 | 0  | -π/2  |
    // | 5     | q5 | 0  | 0  | π/2   |
    // | 6     | q6 | 0  | 0  | 0     |
    println!("Step 1: Building 6 DH matrices individually...");
    let t0 = Instant::now();

    let _t1 = dh_matrix(&q1, &zero, &zero, &half_pi);
    let t1_time = t0.elapsed();
    println!("  T1 built in {:?}", t1_time);

    let _t2 = dh_matrix(&q2, &zero, &a2, &zero);
    let _t3 = dh_matrix(&q3, &zero, &zero, &half_pi);
    let _t4 = dh_matrix(&q4, &d4, &zero, &neg_half_pi);
    let _t5 = dh_matrix(&q5, &zero, &zero, &half_pi);
    let _t6 = dh_matrix(&q6, &zero, &zero, &zero);

    let dh_matrices_time = t0.elapsed();
    println!("  All 6 DH matrices built in {:?}", dh_matrices_time);

    // ── Step 2: FK chain multiplication ──────────────────────────────────
    println!("\nStep 2: Computing FK chain T0_6 = T1·T2·T3·T4·T5·T6...");
    let t0_time = Instant::now();

    let dh_params: [(&Ex, &Ex, &Ex, &Ex); 6] = [
        (&q1, &zero, &zero, &half_pi),
        (&q2, &zero, &a2, &zero),
        (&q3, &zero, &zero, &half_pi),
        (&q4, &d4, &zero, &neg_half_pi),
        (&q5, &zero, &zero, &half_pi),
        (&q6, &zero, &zero, &zero),
    ];
    let fk_total = fk_chain(&dh_params);
    let fk_chain_time = t0_time.elapsed();
    println!("  FK chain computed in {:?}", fk_chain_time);
    assert_eq!(fk_total.shape(), (4, 4));

    // Report size of each FK matrix entry
    let mut max_ops = 0usize;
    let mut max_terms = 0usize;
    for i in 0..4 {
        for j in 0..4 {
            let ops = fk_total.get(i, j).count_ops();
            let terms = fk_total.get(i, j).term_count();
            if ops > max_ops {
                max_ops = ops;
            }
            if terms > max_terms {
                max_terms = terms;
            }
        }
    }
    println!(
        "  FK matrix: total ops={}, total terms={}, max entry ops={}, max entry terms={}",
        matrix_total_ops(&fk_total),
        matrix_total_terms(&fk_total),
        max_ops,
        max_terms
    );

    // ── Step 3: Extract end-effector position ────────────────────────────
    println!("\nStep 3: Extracting end-effector position (px, py, pz)...");
    let t0_pos = Instant::now();
    let px = fk_total.get(0, 3).clone();
    let py = fk_total.get(1, 3).clone();
    let pz = fk_total.get(2, 3).clone();
    let pos_time = t0_pos.elapsed();
    println!("  Position extracted in {:?}", pos_time);
    println!(
        "  px: ops={}, terms={}",
        px.count_ops(),
        px.term_count()
    );
    println!(
        "  py: ops={}, terms={}",
        py.count_ops(),
        py.term_count()
    );
    println!(
        "  pz: ops={}, terms={}",
        pz.count_ops(),
        pz.term_count()
    );

    // ── Step 4: Compute the position Jacobian (3×6) ──────────────────────
    println!("\nStep 4: Computing 3×6 position Jacobian...");
    let t0_jac = Instant::now();

    let jac = jacobian(
        &[&px, &py, &pz],
        &[&q1, &q2, &q3, &q4, &q5, &q6],
    );

    let jac_time = t0_jac.elapsed();
    println!("  Jacobian computed in {:?}", jac_time);
    assert_eq!(jac.shape(), (3, 6));
    println!(
        "  Jacobian: total ops={}, total terms={}",
        matrix_total_ops(&jac),
        matrix_total_terms(&jac)
    );

    // Report per-entry sizes for the Jacobian
    for i in 0..3 {
        for j in 0..6 {
            let ops = jac.get(i, j).count_ops();
            let terms = jac.get(i, j).term_count();
            println!("    J[{i},{j}]: ops={ops:>5}, terms={terms:>4}");
        }
    }

    // ── Step 5: Eval the FK chain ────────────────────────────────────────
    println!("\nStep 5: Evaluating FK at numeric values...");
    let t0_eval = Instant::now();
    let fk_eval = |e: &Ex| -> f64 {
        e.subs(&q1, &ctx.rational(3, 10))
            .subs(&q2, &ctx.rational(5, 10))
            .subs(&q3, &ctx.rational(-2, 10))
            .subs(&q4, &ctx.rational(8, 10))
            .subs(&q5, &ctx.rational(-4, 10))
            .subs(&q6, &ctx.rational(1, 10))
            .subs(&a2, &ctx.rational(4318, 10000))
            .subs(&d4, &ctx.rational(4331, 10000))
            .eval()
            .eval_f64()
            .unwrap()
    };
    let px_num = fk_eval(&px);
    let py_num = fk_eval(&py);
    let pz_num = fk_eval(&pz);
    let eval_time = t0_eval.elapsed();
    println!("  Eval completed in {:?}", eval_time);
    println!("  End-effector position: ({px_num:.6}, {py_num:.6}, {pz_num:.6})");

    // Sanity check: the position should be reasonable (not NaN/Inf)
    assert!(px_num.is_finite(), "px must be finite");
    assert!(py_num.is_finite(), "py must be finite");
    assert!(pz_num.is_finite(), "pz must be finite");
    println!("  ✓ Position is finite");

    // ── Step 6: Eval the FK chain via fk_position ────────────────────────
    println!("\nStep 6: Cross-check with fk_position()...");
    let t0_fkp = Instant::now();
    let (fpx, fpy, fpz) = fk_position(&dh_params);
    let fkp_time = t0_fkp.elapsed();
    println!("  fk_position() computed in {:?}", fkp_time);

    let fpx_num = fk_eval(&fpx);
    let fpy_num = fk_eval(&fpy);
    let fpz_num = fk_eval(&fpz);
    assert_near(px_num, fpx_num, 1e-10, "px from chain vs fk_position");
    assert_near(py_num, fpy_num, 1e-10, "py from chain vs fk_position");
    assert_near(pz_num, fpz_num, 1e-10, "pz from chain vs fk_position");
    println!("  ✓ fk_position matches manual chain extraction");

    // ── Step 7: Simplify a Jacobian entry ────────────────────────────────
    println!("\nStep 7: Simplifying J[0,0] with trigsimp...");
    let t0_trig = Instant::now();
    let j00_simplified = jac.get(0, 0).simplify_trig();
    let trig_jac_time = t0_trig.elapsed();
    println!("  trigsimp(J[0,0]) in {:?}", trig_jac_time);
    println!(
        "  Before: ops={}, terms={}",
        jac.get(0, 0).count_ops(),
        jac.get(0, 0).term_count()
    );
    println!(
        "  After:  ops={}, terms={}",
        j00_simplified.count_ops(),
        j00_simplified.term_count()
    );

    // ── Step 8: Try eval on the FK matrix ────────────────────────────────
    println!("\nStep 8: Evaluating expanded FK entries...");
    let t0_expand = Instant::now();
    let px_expanded = px.expand();
    let expand_time = t0_expand.elapsed();
    println!(
        "  px.expand() in {:?}, ops={}, terms={}",
        expand_time,
        px_expanded.count_ops(),
        px_expanded.term_count()
    );

    // ── Step 9: Code generation for the Jacobian ─────────────────────────
    println!("\nStep 9: Generating Rust code for the 3×6 Jacobian...");
    let t0_codegen = Instant::now();

    let code_result = jac.to_rust_fn(
        "puma_jacobian",
        &["q1", "q2", "q3", "q4", "q5", "q6", "a2", "d4"],
    );

    let codegen_time = t0_codegen.elapsed();
    match &code_result {
        Ok(code) => {
            println!("  Codegen completed in {:?}", codegen_time);
            println!("  Generated code length: {} bytes", code.len());
            // Print the first few lines
            let lines: Vec<&str> = code.lines().collect();
            println!("  First 5 lines:");
            for line in lines.iter().take(5) {
                println!("    {line}");
            }
            println!("  ... ({} total lines)", lines.len());
        }
        Err(e) => {
            println!("  Codegen FAILED in {:?}: {e}", codegen_time);
        }
    }

    // ── Step 10: Code generation for the FK position ─────────────────────
    println!("\nStep 10: Generating Rust code for FK position (3 expressions)...");
    let t0_codegen2 = Instant::now();
    let pos_matrix = Matrix::new(vec![vec![px.clone(), py.clone(), pz.clone()]]).unwrap();
    let pos_code_result = pos_matrix.to_rust_fn(
        "puma_fk_position",
        &["q1", "q2", "q3", "q4", "q5", "q6", "a2", "d4"],
    );
    let codegen2_time = t0_codegen2.elapsed();
    match &pos_code_result {
        Ok(code) => {
            println!("  Position codegen completed in {:?}", codegen2_time);
            println!("  Generated code length: {} bytes", code.len());
            let lines: Vec<&str> = code.lines().collect();
            println!("  Total lines: {}", lines.len());
        }
        Err(e) => {
            println!("  Position codegen FAILED in {:?}: {e}", codegen2_time);
        }
    }

    // ── Step 11: Try 6-DOF dynamics (kinetic energy for first 3 joints) ──
    println!("\nStep 11: Attempting simplified 6-DOF dynamics (first 3 joints only)...");

    let qd1 = ctx.symbol("qd1");
    let qd2 = ctx.symbol("qd2");
    let qd3 = ctx.symbol("qd3");

    let m1 = ctx.symbol("m1");
    let m2 = ctx.symbol("m2");
    let m3 = ctx.symbol("m3");

    // Use the first 3 joints' FK positions for a simplified dynamics test
    let dh_1 = [(&q1, &zero, &zero, &half_pi)];
    let dh_2 = [
        (&q1, &zero, &zero, &half_pi),
        (&q2, &zero, &a2, &zero),
    ];
    let dh_3 = [
        (&q1, &zero, &zero, &half_pi),
        (&q2, &zero, &a2, &zero),
        (&q3, &zero, &zero, &half_pi),
    ];

    let t0_dyn = Instant::now();

    let (_, _p1y, p1z) = fk_position(&dh_1);
    let (_, _p2y, p2z) = fk_position(&dh_2);
    let (_, _p3y, p3z) = fk_position(&dh_3);

    // For a simplified spatial arm, potential energy uses z component
    // (assuming gravity is along -z)
    let g_sym = ctx.symbol("g");
    let pe_6 = &(&(&m1 * &g_sym) * &p1z)
        + &(&(&(&m2 * &g_sym) * &p2z) + &(&(&m3 * &g_sym) * &p3z));

    let dyn_setup_time = t0_dyn.elapsed();
    println!("  6-DOF potential energy built in {:?}", dyn_setup_time);
    println!(
        "  V ops={}, terms={}",
        pe_6.count_ops(),
        pe_6.term_count()
    );

    // Build kinetic energy using the position Jacobian approach
    // T = ½ Σ mᵢ Jᵢᵀ Jᵢ integrated over qdot
    // For simplicity, compute velocity components by differentiating positions
    let t0_ke6 = Instant::now();

    let q_vars_3 = [q1.clone(), q2.clone(), q3.clone()];
    let (p1x, p1y, p1z) = fk_position(&dh_1);
    let (p2x, p2y, p2z) = fk_position(&dh_2);
    let (p3x, p3y, p3z) = fk_position(&dh_3);

    // Velocity of each COM = J_i * qdot
    // v_ix = Σ_j (∂p_ix/∂q_j) * qd_j, etc.
    let qdot_vars = [&qd1, &qd2, &qd3];
    let half = ctx.rational(1, 2);

    let compute_link_ke = |px: &Ex, py: &Ex, pz: &Ex, mass: &Ex| -> Ex {
        let mut vx = ctx.int(0);
        let mut vy = ctx.int(0);
        let mut vz = ctx.int(0);
        for (k, qk) in q_vars_3.iter().enumerate() {
            vx = &vx + &(&px.diff(qk) * qdot_vars[k]);
            vy = &vy + &(&py.diff(qk) * qdot_vars[k]);
            vz = &vz + &(&pz.diff(qk) * qdot_vars[k]);
        }
        let v_sq = &(&(&vx * &vx) + &(&vy * &vy)) + &(&vz * &vz);
        &(&half * mass) * &v_sq
    };

    let ke1 = compute_link_ke(&p1x, &p1y, &p1z, &m1);
    let ke2 = compute_link_ke(&p2x, &p2y, &p2z, &m2);
    let ke3 = compute_link_ke(&p3x, &p3y, &p3z, &m3);
    let ke_6 = &(&ke1 + &ke2) + &ke3;

    let ke6_time = t0_ke6.elapsed();
    println!("  Spatial KE (3 joints) built in {:?}", ke6_time);
    println!(
        "  T ops={}, terms={}",
        ke_6.count_ops(),
        ke_6.term_count()
    );

    // Expand before mass_matrix extraction
    let t0_expand6 = Instant::now();
    let ke_6_expanded = ke_6.expand();
    let expand6_time = t0_expand6.elapsed();
    println!(
        "  T expanded in {:?}, ops={}, terms={}",
        expand6_time,
        ke_6_expanded.count_ops(),
        ke_6_expanded.term_count()
    );

    // Mass matrix for the 3-joint spatial arm
    let t0_mm6 = Instant::now();
    let mm_6 = mass_matrix(&ke_6_expanded, &[&qd1, &qd2, &qd3]);
    let mm6_time = t0_mm6.elapsed();
    println!(
        "  mass_matrix() for spatial 3-DOF in {:?}",
        mm6_time
    );
    assert_eq!(mm_6.shape(), (3, 3));
    println!(
        "  M total ops={}, total terms={}",
        matrix_total_ops(&mm_6),
        matrix_total_terms(&mm_6)
    );

    // ── Summary ──────────────────────────────────────────────────────────
    println!("\n{}", "─".repeat(60));
    println!("  6-DOF TIMING SUMMARY");
    println!("{}", "─".repeat(60));
    println!("  DH matrices (6):      {:>10?}", dh_matrices_time);
    println!("  FK chain (6 joints):  {:>10?}", fk_chain_time);
    println!("  Position extract:     {:>10?}", pos_time);
    println!("  Jacobian (3×6):       {:>10?}", jac_time);
    println!("  Eval FK:              {:>10?}", eval_time);
    println!("  fk_position():        {:>10?}", fkp_time);
    println!("  trigsimp(J[0,0]):     {:>10?}", trig_jac_time);
    println!("  expand(px):           {:>10?}", expand_time);
    println!("  Codegen Jacobian:     {:>10?}", codegen_time);
    println!("  Codegen position:     {:>10?}", codegen2_time);
    println!("  ── Spatial 3-DOF dynamics ──");
    println!("  PE setup:             {:>10?}", dyn_setup_time);
    println!("  KE build (3 joints):  {:>10?}", ke6_time);
    println!("  KE expand:            {:>10?}", expand6_time);
    println!("  mass_matrix (3-DOF):  {:>10?}", mm6_time);
    let total_6dof = dh_matrices_time
        + fk_chain_time
        + pos_time
        + jac_time
        + eval_time
        + fkp_time
        + trig_jac_time
        + expand_time
        + codegen_time
        + codegen2_time
        + dyn_setup_time
        + ke6_time
        + expand6_time
        + mm6_time;
    println!("  ──────────────────────────────");
    println!("  TOTAL:                {:>10?}", total_6dof);
    println!();

    // Final pass/fail verdict
    if codegen_time.as_secs() > 120 {
        println!("  ⚠ BOTTLENECK: Codegen is very slow (>120s)");
    }
    if jac_time.as_secs() > 60 {
        println!("  ⚠ BOTTLENECK: Jacobian computation is very slow (>60s)");
    }
    if fk_chain_time.as_secs() > 60 {
        println!("  ⚠ BOTTLENECK: FK chain is very slow (>60s)");
    }
    if mm6_time.as_secs() > 60 {
        println!("  ⚠ BOTTLENECK: Spatial mass_matrix is very slow (>60s)");
    }
    if total_6dof.as_secs() < 60 {
        println!("  ✓ All 6-DOF operations completed in under 60 seconds");
    }
}
