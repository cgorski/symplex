//! Lagrangian Dynamics — derive equations of motion for mechanical systems.
//!
//! Demonstrates:
//! - Simple pendulum: Euler–Lagrange equations, mass matrix, gravity vector
//! - Double pendulum (2-DOF): full manipulator equation M(q)q̈ + C(q,q̇)q̇ + g(q) = τ
//! - Mass matrix symmetry check
//! - Numerical evaluation at specific configurations
//!
//! Run with: cargo run --example dynamics

use symplex::dynamics::*;
use symplex::prelude::*;

fn main() {
    println!("=== Lagrangian Dynamics ===\n");

    // ════════════════════════════════════════════════════════════════
    // Part 1: Simple Pendulum (1-DOF)
    // ════════════════════════════════════════════════════════════════

    println!("--- Simple Pendulum (1-DOF) ---\n");

    let ctx = Context::new();
    symplex::syms!(ctx; q, qd, qdd);
    let m = ctx.symbol("m");
    let l = ctx.symbol("L");
    let g = ctx.symbol("g");

    // Simple pendulum: T = ½·m·L²·q̇², V = -m·g·L·cos(q)
    //
    // ctx.rational(1, 2) returns the exact rational 1/2, avoiding any
    // floating-point approximation in the kinetic energy expression.
    let half = ctx.rational(1, 2);
    let ke = &half * &m * &l.powi(2) * &qd.powi(2);
    let neg_m = -&m;
    let pe = &neg_m * &g * &l * &q.cos();

    println!("T = {ke}");
    println!("V = {pe}");

    // Euler-Lagrange equations: d/dt(∂L/∂q̇) - ∂L/∂q = τ
    let eqs = euler_lagrange(&ke, &pe, &[(&q, &qd)], &[&qdd]);
    println!("\nEquation of motion:");
    println!("  τ = {}", eqs[0]);

    // Mass matrix: M_ij = ∂²T / ∂q̇ᵢ∂q̇ⱼ
    let mm = mass_matrix(&ke, &[&qd]);
    println!("\nMass matrix: {}", mm.get(0, 0));

    // Gravity vector: gᵢ = ∂V/∂qᵢ
    let gv = gravity_vector(&pe, &[&q]);
    println!("Gravity: {}", gv[0]);

    // Numerical evaluation at rest with unit acceleration
    // m=1, L=1, g=10, q=0 (hanging straight down), qd=0, qdd=1
    //
    // Why τ = 1.00:
    //   τ = m·L²·q̈ + m·g·L·sin(q)
    // At q = 0, sin(0) = 0, so the gravity term vanishes entirely.
    // Only the inertial term remains: τ = 1·1²·1 = 1.00.
    let tau = eqs[0]
        .eval_f64_with(&[(&m, 1), (&l, 1), (&g, 10), (&q, 0), (&qd, 0), (&qdd, 1)])
        .unwrap();
    println!("\nτ at rest with unit acceleration: {tau:.2}");

    // Evaluate at q = π/4 (45°) — now gravity contributes
    // τ = m·L²·q̈ + m·g·L·sin(π/4) = 1 + 10·sin(π/4) ≈ 1 + 7.071 = 8.071
    let tau_45 = eqs[0]
        .eval_f64_with(&[(&m, 1), (&l, 1), (&g, 10), (&q, 1), (&qd, 0), (&qdd, 1)])
        .unwrap();
    println!("τ at q≈1 rad, unit accel: {tau_45:.4}");

    // ════════════════════════════════════════════════════════════════
    // Part 2: Double Pendulum (2-DOF)
    // ════════════════════════════════════════════════════════════════

    println!("\n\n--- Double Pendulum (2-DOF) ---\n");

    symplex::syms!(ctx; q1, q2, qd1, qd2, qdd1, qdd2);
    let m1 = ctx.symbol("m1");
    let m2 = ctx.symbol("m2");
    let l1 = ctx.symbol("L1");
    let l2 = ctx.symbol("L2");

    // Double pendulum kinetic energy:
    //   T = ½·m1·L1²·q̇1²
    //     + ½·m2·(L1²·q̇1² + L2²·q̇2² + 2·L1·L2·q̇1·q̇2·cos(q1-q2))
    //
    // This is the standard form where both masses are point masses
    // at the end of each link.

    let ke_1 = &half * &m1 * &l1.powi(2) * &qd1.powi(2);

    let ke_2_term1 = &half * &m2 * &l1.powi(2) * &qd1.powi(2);
    let ke_2_term2 = &half * &m2 * &l2.powi(2) * &qd2.powi(2);
    let q_diff = &q1 - &q2;
    let ke_2_term3 = &m2 * &l1 * &l2 * &qd1 * &qd2 * &q_diff.cos();
    let ke_double = &(&(&ke_1 + &ke_2_term1) + &ke_2_term2) + &ke_2_term3;

    println!("T (double pendulum):");
    println!("  {ke_double}");

    // Double pendulum potential energy:
    //   V = -(m1+m2)·g·L1·cos(q1) - m2·g·L2·cos(q2)
    let neg_m1_plus_m2 = -&(&m1 + &m2);
    let pe_1 = &neg_m1_plus_m2 * &g * &l1 * &q1.cos();
    let neg_m2 = -&m2;
    let pe_2 = &neg_m2 * &g * &l2 * &q2.cos();
    let pe_double = &pe_1 + &pe_2;

    println!("\nV (double pendulum):");
    println!("  {pe_double}");

    // Euler-Lagrange equations
    let eqs_double = euler_lagrange(
        &ke_double,
        &pe_double,
        &[(&q1, &qd1), (&q2, &qd2)],
        &[&qdd1, &qdd2],
    );
    println!("\nEquation of motion (joint 1):");
    println!("  τ₁ = {}", eqs_double[0]);
    println!("\nEquation of motion (joint 2):");
    println!("  τ₂ = {}", eqs_double[1]);

    // ── Mass matrix (2×2) ──────────────────────────────────────────
    let mm_double = mass_matrix(&ke_double, &[&qd1, &qd2]);
    println!("\nMass matrix M(q):");
    println!("  M[0,0] = {}", mm_double.get(0, 0));
    println!("  M[0,1] = {}", mm_double.get(0, 1));
    println!("  M[1,0] = {}", mm_double.get(1, 0));
    println!("  M[1,1] = {}", mm_double.get(1, 1));

    // ── Symmetry check ─────────────────────────────────────────────
    //
    // The mass matrix for any physical system must be symmetric:
    //   M[i,j] = M[j,i]
    //
    // We check this by comparing the off-diagonal entries.
    let is_sym = mm_double.is_symmetric();
    println!("\nMass matrix symmetric? {is_sym:?}");

    // ── Gravity vector ─────────────────────────────────────────────
    let gv_double = gravity_vector(&pe_double, &[&q1, &q2]);
    println!("\nGravity vector:");
    println!("  g₁ = {}", gv_double[0]);
    println!("  g₂ = {}", gv_double[1]);

    // ── Coriolis matrix ────────────────────────────────────────────
    let coriolis = coriolis_matrix(&mm_double, &[&q1, &q2], &[&qd1, &qd2]);
    println!("\nCoriolis matrix C(q, q̇):");
    println!("  C[0,0] = {}", coriolis.get(0, 0));
    println!("  C[0,1] = {}", coriolis.get(0, 1));
    println!("  C[1,0] = {}", coriolis.get(1, 0));
    println!("  C[1,1] = {}", coriolis.get(1, 1));

    // ── Full manipulator equation via convenience function ─────────
    let (mass, cor, grav) =
        manipulator_equation(&ke_double, &pe_double, &[&q1, &q2], &[&qd1, &qd2]);
    println!("\nFull manipulator equation: M(q)q̈ + C(q,q̇)q̇ + g(q) = τ");
    println!("  M shape: {:?}", mass.shape());
    println!("  C shape: {:?}", cor.shape());
    println!("  g length: {}", grav.len());

    // ── Christoffel symbols ────────────────────────────────────────
    let christoffel = christoffel_symbols(&mm_double, &[&q1, &q2]);
    println!("\nChristoffel symbols (Γ_ijk):");
    for (i, plane) in christoffel.iter().enumerate().take(2) {
        for (j, row) in plane.iter().enumerate().take(2) {
            for (k, gamma) in row.iter().enumerate().take(2) {
                let s = format!("{gamma}");
                if s != "0" {
                    println!("  Γ_{{{i}{j}{k}}} = {s}");
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Part 3: Numerical Evaluation at Specific Configuration
    // ════════════════════════════════════════════════════════════════

    println!("\n\n--- Numerical Evaluation ---\n");

    // Configuration: q1=0, q2=0 (both links hanging straight down)
    // m1=1, m2=1, L1=1, L2=1, g=9.81
    // qd1=0, qd2=0 (at rest), qdd1=1, qdd2=0

    println!("Config: q=(0,0), q̇=(0,0), q̈=(1,0)");
    println!("Params: m1=1, m2=1, L1=1, L2=1");

    // Evaluate mass matrix numerically (using g=10 integer approximation)
    let subs_no_g: &[(&symplex::prelude::Ex, i64)] =
        &[(&m1, 1), (&m2, 1), (&l1, 1), (&l2, 1), (&q1, 0), (&q2, 0)];

    let m00 = mm_double.get(0, 0).eval_f64_with(subs_no_g);
    let m01 = mm_double.get(0, 1).eval_f64_with(subs_no_g);
    let m10 = mm_double.get(1, 0).eval_f64_with(subs_no_g);
    let m11 = mm_double.get(1, 1).eval_f64_with(subs_no_g);
    println!("\nNumerical mass matrix at q=(0,0):");
    println!(
        "  M = [[{:.4}, {:.4}],",
        m00.unwrap_or(f64::NAN),
        m01.unwrap_or(f64::NAN)
    );
    println!(
        "       [{:.4}, {:.4}]]",
        m10.unwrap_or(f64::NAN),
        m11.unwrap_or(f64::NAN)
    );

    // At q1=q2=0, cos(q1-q2) = cos(0) = 1, so:
    //   M[0,0] = (m1+m2)·L1² = 2
    //   M[0,1] = M[1,0] = m2·L1·L2·cos(0) = 1
    //   M[1,1] = m2·L2² = 1

    // Evaluate the EOM torques at the test configuration
    let tau1 = eqs_double[0].eval_f64_with(&[
        (&m1, 1),
        (&m2, 1),
        (&l1, 1),
        (&l2, 1),
        (&g, 10),
        (&q1, 0),
        (&q2, 0),
        (&qd1, 0),
        (&qd2, 0),
        (&qdd1, 1),
        (&qdd2, 0),
    ]);
    let tau2 = eqs_double[1].eval_f64_with(&[
        (&m1, 1),
        (&m2, 1),
        (&l1, 1),
        (&l2, 1),
        (&g, 10),
        (&q1, 0),
        (&q2, 0),
        (&qd1, 0),
        (&qd2, 0),
        (&qdd1, 1),
        (&qdd2, 0),
    ]);

    println!("\nTorques at test config (g=10):");
    println!("  τ₁ = {:.4}", tau1.unwrap_or(f64::NAN));
    println!("  τ₂ = {:.4}", tau2.unwrap_or(f64::NAN));

    // At q=(0,0), qd=(0,0), qdd=(1,0):
    //   τ₁ = M[0,0]·qdd1 + M[0,1]·qdd2 + gravity terms
    //       = 2·1 + 1·0 + 0 = 2  (sin(0)=0, so gravity vanishes)
    //   τ₂ = M[1,0]·qdd1 + M[1,1]·qdd2 + gravity terms
    //       = 1·1 + 1·0 + 0 = 1

    // ── Determinant of mass matrix ─────────────────────────────────
    let det = mm_double.det().unwrap();
    println!("\ndet(M) = {det}");
    let det_val = det.eval_f64_with(subs_no_g);
    println!("det(M) at q=(0,0): {:.4}", det_val.unwrap_or(f64::NAN));
    println!("  (Positive definite mass matrix has positive determinant)");

    // ── Total time derivative ──────────────────────────────────────
    println!("\n--- Total Time Derivative ---");
    // d/dt(q1) = qd1
    let dt_q1 = total_time_derivative(&q1, &[(&q1, &qd1), (&q2, &qd2)], &[&qdd1, &qdd2]);
    let dt_q1_val = dt_q1.subs(&qd1, &ctx.int(7)).subs(&qd2, &ctx.int(0)).eval();
    println!("d/dt(q1) = {dt_q1}");
    println!("  at qd1=7: {dt_q1_val}");

    println!("\n✓ Done!");
}
