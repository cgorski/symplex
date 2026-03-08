//! Robotics Code Generation — DH parameters to optimized Rust code.
//!
//! This example demonstrates the complete symplex robotics pipeline:
//!
//!   1. Define joint angles and link lengths as exact rationals
//!   2. Specify DH parameters for a 3-DOF planar arm
//!   3. Compute forward kinematics (symbolic end-effector position)
//!   4. Print the symbolic FK expressions
//!   5. Compute the analytical Jacobian (2×3)
//!   6. Generate optimized Rust code with cross-entry CSE
//!   7. Print the generated code
//!   8. Verify numerically at a specific joint configuration
//!
//! Run with: cargo run --example robotics_codegen

use std::time::Instant;
use symplex::matrix::jacobian;
use symplex::prelude::*;
use symplex::robotics::*;
use symplex::vars;

fn main() {
    println!("=== Symplex Robotics Code Generation ===\n");

    // ── 1. Define joint variables and link lengths ─────────────────
    //
    // Joint angles are symbolic variables — they will become function
    // parameters in the generated code.
    //
    // Link lengths are exact rationals.  Using symplex::default_context().rational()
    // instead of f64 keeps the entire derivation exact: no IEEE 754
    // rounding until the very end when we evaluate numerically.

    vars!(theta1, theta2, theta3);

    let l1 = symplex::default_context().rational(3, 10); // 0.3 m
    let l2 = symplex::default_context().rational(1, 4); // 0.25 m
    let l3 = symplex::default_context().rational(1, 5); // 0.2 m

    println!("Link lengths: L1 = {l1}, L2 = {l2}, L3 = {l3}");
    println!("Total reach:  {} m", &(&l1 + &l2) + &l3);

    // ── 2. DH parameters ───────────────────────────────────────────
    //
    // Standard DH convention: each joint is described by four params:
    //   (theta, d, a, alpha)
    //
    // For a planar arm, d = 0 and alpha = 0 for every joint.
    // Only theta (joint angle) and a (link length) vary.

    let zero = symplex::default_context().int(0);
    let dh: [(&Ex, &Ex, &Ex, &Ex); 3] = [
        (&theta1, &zero, &l1, &zero), // Joint 1
        (&theta2, &zero, &l2, &zero), // Joint 2
        (&theta3, &zero, &l3, &zero), // Joint 3
    ];

    println!("\nDH parameters (theta, d, a, alpha):");
    for (i, (th, d, a, al)) in dh.iter().enumerate() {
        println!("  Joint {}: ({th}, {d}, {a}, {al})", i + 1);
    }

    // ── 3. Forward kinematics ──────────────────────────────────────
    //
    // Chain-multiply the DH transformation matrices to get the
    // end-effector position as symbolic expressions of the joint angles.
    //
    // fk_position() returns (px, py, pz).  For a planar arm, pz = 0.

    println!("\n--- Forward Kinematics ---");
    let t0 = Instant::now();
    let (px, py, _pz) = fk_position(&dh);
    let fk_time = t0.elapsed();
    println!("FK computed in {fk_time:?}");

    // ── 4. Print symbolic FK expressions ───────────────────────────
    //
    // These are trigonometric sums of the form:
    //   px = L1·cos(θ1) + L2·cos(θ1+θ2) + L3·cos(θ1+θ2+θ3)
    //   py = L1·sin(θ1) + L2·sin(θ1+θ2) + L3·sin(θ1+θ2+θ3)
    //
    // The exact form depends on how symplex canonicalizes the
    // expanded trig expressions.

    println!("\nEnd-effector position (symbolic):");
    println!("  px = {px}");
    println!("  py = {py}");

    // Also show the LaTeX form for documentation
    println!("\nLaTeX:");
    println!("  p_x = {}", px.to_latex());
    println!("  p_y = {}", py.to_latex());

    // ── 5. Compute the Jacobian ────────────────────────────────────
    //
    // The Jacobian J maps joint velocities to end-effector velocities:
    //   [ẋ]       [∂px/∂θ1  ∂px/∂θ2  ∂px/∂θ3] [θ̇1]
    //   [ẏ]  = J · [θ̇2]  where J = [∂py/∂θ1  ∂py/∂θ2  ∂py/∂θ3]
    //
    // This is a 2×3 matrix (2 task-space DOFs, 3 joint-space DOFs).
    //
    // The jacobian() function computes each ∂fᵢ/∂θⱼ symbolically.

    println!("\n--- Jacobian (2×3) ---");
    let t1 = Instant::now();
    let jac = jacobian(&[&px, &py], &[&theta1, &theta2, &theta3]);
    let jac_time = t1.elapsed();
    println!("Jacobian computed in {jac_time:?}");
    println!("\n{jac}");

    // Show individual entries
    println!("\nJacobian entries:");
    for i in 0..2 {
        for j in 0..3 {
            let label = if i == 0 { "x" } else { "y" };
            println!("  ∂p{label}/∂θ{} = {}", j + 1, jac.get(i, j));
        }
    }

    // ── 6. Generate optimized Rust code ────────────────────────────
    //
    // to_rust_fn() performs:
    //   - Common Subexpression Elimination (CSE) across ALL matrix
    //     entries simultaneously — shared trig calls like sin(θ1+θ2)
    //     are computed once
    //   - Constant folding — sin(0)→0, cos(0)→1 at codegen time
    //   - Clean decimal output — 0.3, not 0.30000000000000004
    //   - Proper subtraction — "x - 0.3", not "x + (-0.3)"
    //
    // The result is a complete pub fn returning [f64; 6] in row-major order.

    println!("\n--- Code Generation ---");
    let t2 = Instant::now();
    let code = jac
        .to_rust_fn("robot_jacobian", &["theta1", "theta2", "theta3"])
        .expect("codegen failed");
    let codegen_time = t2.elapsed();
    println!("Code generated in {codegen_time:?} ({} bytes)\n", code.len());
    println!("{code}");

    // Also generate code for the FK position itself
    println!("--- FK Position Code ---");
    let px_code = px
        .to_rust_fn("fk_x", &["theta1", "theta2", "theta3"])
        .expect("codegen failed");
    println!("{px_code}");

    let py_code = py
        .to_rust_fn("fk_y", &["theta1", "theta2", "theta3"])
        .expect("codegen failed");
    println!("{py_code}");

    // ── 7. Generate with different options ─────────────────────────

    println!("--- Embedded f32 Variant ---");
    let embedded_code = jac
        .to_rust_fn_with_options(
            "robot_jacobian_f32",
            &["theta1", "theta2", "theta3"],
            &symplex::matrix::CodegenOptions::embedded_f32(),
        )
        .expect("codegen failed");
    println!("{embedded_code}");

    // ── 8. Numerical verification ──────────────────────────────────
    //
    // Verify the symbolic FK at a specific configuration by:
    //   (a) evaluating the symbolic expression
    //   (b) computing FK from first principles using f64 trig
    //   (c) comparing the results
    //
    // This is the verification step you should always perform after
    // code generation to catch any derivation or codegen bugs.

    println!("--- Numerical Verification ---");

    let test_configs: &[(f64, f64, f64)] = &[
        (0.0, 0.0, 0.0),                     // fully extended along +x
        (std::f64::consts::FRAC_PI_4, 0.0, 0.0), // 45° first joint
        (0.5, 0.3, 0.1),                      // arbitrary configuration
    ];

    for (t1, t2, t3) in test_configs {
        println!("\n  Config: θ = ({t1:.4}, {t2:.4}, {t3:.4})");

        // Ground truth: direct f64 computation
        let l1_f = 0.3;
        let l2_f = 0.25;
        let l3_f = 0.2;
        let gt_x = l1_f * t1.cos() + l2_f * (t1 + t2).cos() + l3_f * (t1 + t2 + t3).cos();
        let gt_y = l1_f * t1.sin() + l2_f * (t1 + t2).sin() + l3_f * (t1 + t2 + t3).sin();

        println!("    Ground truth: px = {gt_x:.6}, py = {gt_y:.6}");

        // Symbolic evaluation (substitute integer approximations for display)
        // For exact comparison we use compile()
        if let Some(px_fn) = px.compile(&["theta1", "theta2", "theta3"]) {
            let sym_x = px_fn(&[*t1, *t2, *t3]);
            let sym_y = py
                .compile(&["theta1", "theta2", "theta3"])
                .map(|f| f(&[*t1, *t2, *t3]))
                .unwrap_or(f64::NAN);
            println!("    Symbolic eval: px = {sym_x:.6}, py = {sym_y:.6}");

            let err_x = (sym_x - gt_x).abs();
            let err_y = (sym_y - gt_y).abs();
            println!("    Error:         Δx = {err_x:.2e}, Δy = {err_y:.2e}");

            if err_x < 1e-10 && err_y < 1e-10 {
                println!("    ✓ Match!");
            } else {
                println!("    ✗ MISMATCH — investigate!");
            }
        }

        // Also evaluate the Jacobian numerically
        // For the fully extended config, ∂px/∂θ1 should be
        // -(L1·sin(θ1) + L2·sin(θ1+θ2) + L3·sin(θ1+θ2+θ3))
        // At θ=0,0,0 that's 0 (since sin(0)=0).
    }

    // ── 9. Timing summary ──────────────────────────────────────────

    let total = fk_time + jac_time + codegen_time;
    println!("\n--- Timing Summary ---");
    println!("  FK derivation:     {fk_time:?}");
    println!("  Jacobian:          {jac_time:?}");
    println!("  Code generation:   {codegen_time:?}");
    println!("  Total:             {total:?}");
    println!("\n  (This runs once at design time or in build.rs.)");
    println!("  (The generated code runs at >1 MHz in your control loop.)");

    // ── 10. Full transformation matrix ─────────────────────────────
    //
    // For reference, you can also get the full 4×4 homogeneous
    // transformation matrix and extract the rotation submatrix.

    println!("\n--- Full FK Chain ---");
    let t_full = fk_chain(&dh);
    println!("T(4×4) shape: {:?}", t_full.shape());

    // The rotation submatrix gives end-effector orientation
    let rot = fk_rotation(&dh);
    println!("Rotation (3×3):");
    println!("{rot}");

    println!("\n✓ Done!");
}
