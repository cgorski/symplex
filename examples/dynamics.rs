//! Lagrangian Dynamics — derive equations of motion for a pendulum.
//!
//! Run with: cargo run --example dynamics

use symplex::dynamics::*;
use symplex::vars;

fn main() {
    println!("=== Lagrangian Dynamics ===\n");

    vars!(q, qd, qdd);
    let m = symplex::var("m");
    let l = symplex::var("L");
    let g = symplex::var("g");

    // Simple pendulum: T = ½·m·L²·q̇², V = -m·g·L·cos(q)
    let half = symplex::half();
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
    // m=1, L=1, g=10, q=0, qd=0, qdd=1
    let tau = eqs[0]
        .eval_f64_with(&[(&m, 1), (&l, 1), (&g, 10), (&q, 0), (&qd, 0), (&qdd, 1)])
        .unwrap();
    println!("\nτ at rest with unit acceleration: {tau:.2}");

    println!("\n✓ Done!");
}
