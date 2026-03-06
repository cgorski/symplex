//! Robotics Code Generation — DH parameters to optimized Rust code.
//!
//! Derives the Jacobian for a 3-DOF planar robot arm and generates
//! optimized numerical code with common subexpression elimination.
//!
//! Run with: cargo run --example robotics_codegen

use symplex::matrix::jacobian;
use symplex::prelude::*;
use symplex::robotics::*;
use symplex::vars;
use std::time::Instant;

fn main() {
    println!("=== Symplex Robotics Code Generation ===\n");

    // Define joint angles and link lengths
    vars!(theta1, theta2, theta3);
    let l1 = symplex::rational(3, 10); // 0.3m
    let l2 = symplex::rational(1, 4); // 0.25m
    let l3 = symplex::rational(1, 5); // 0.2m

    // Define DH parameters for a 3-DOF planar arm
    // Each joint: (theta, d, a, alpha) — all alpha=0 for planar
    let zero = symplex::int(0);
    let dh: [(&Ex, &Ex, &Ex, &Ex); 3] = [
        (&theta1, &zero, &l1, &zero),
        (&theta2, &zero, &l2, &zero),
        (&theta3, &zero, &l3, &zero),
    ];

    // Compute forward kinematics
    let t0 = Instant::now();
    let (px, py, _pz) = fk_position(&dh);
    println!("FK position computed in {:?}", t0.elapsed());
    println!("  x = {px}");
    println!("  y = {py}");

    // Compute the Jacobian
    let t1 = Instant::now();
    let jac = jacobian(
        &[&px, &py],
        &[&theta1, &theta2, &theta3],
    );
    println!("\nJacobian (2x3) computed in {:?}", t1.elapsed());
    for i in 0..jac.nrows() {
        for j in 0..jac.ncols() {
            println!("  J[{i}][{j}] = {}", jac.get(i, j));
        }
    }

    // Generate optimized Rust code with cross-entry CSE
    let t2 = Instant::now();
    let code = jac
        .to_rust_fn("robot_jacobian", &["theta1", "theta2", "theta3"])
        .expect("code generation should succeed");
    println!("\nRust code generated in {:?}", t2.elapsed());
    println!("Generated code ({} bytes):\n", code.len());
    println!("{code}");

    println!("✓ Pipeline complete: DH → FK → Jacobian → Optimized Rust Code");
}
