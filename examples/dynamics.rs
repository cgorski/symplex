//! Lagrangian Dynamics — derive equations of motion for a pendulum.
//!
//! Uses the Euler–Lagrange formulation to symbolically derive the equation
//! of motion for a simple pendulum, then evaluates the generalized force
//! (torque) at a specific configuration.
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
    //
    // symplex::half() returns the exact rational 1/2, avoiding any
    // floating-point approximation in the kinetic energy expression.
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

    println!("\n✓ Done!");
}
