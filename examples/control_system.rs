//! Control System Analysis — state-space models and stability.
//!
//! Analyzes a mass-spring-damper system using state-space representation,
//! checks stability, and computes transfer function properties.
//!
//! Run with: cargo run --example control_system

use symplex::control::*;
use symplex::prelude::*;
use symplex::vars;

fn main() {
    println!("=== Control System Analysis ===\n");

    vars!(s);

    // Mass-spring-damper: mẍ + cẋ + kx = F
    // State-space form with k/m=4, c/m=3
    let a = matrix![[0, 1], [-4, -3]];
    let b = matrix![[0], [1]];
    let c = matrix![[1, 0]];
    let d = matrix![[0]];

    let sys = StateSpace::new(a, b, c, d);
    println!(
        "States: {}, Inputs: {}, Outputs: {}",
        sys.num_states(),
        sys.num_inputs(),
        sys.num_outputs()
    );

    // Poles (eigenvalues)
    let poles = sys.poles(&s);
    println!(
        "Poles: {:?}",
        poles.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
    );

    // Stability
    match sys.is_stable() {
        Some(true) => println!("Stable: yes"),
        Some(false) => println!("Stable: no"),
        None => println!("Stable: undetermined"),
    }
    println!("Controllable: {}", sys.is_controllable());
    println!("Observable: {}", sys.is_observable());

    // Transfer function: G(s) = 1 / (s² + 3s + 4)
    let tf = TransferFunction::from_coeffs(&[1], &[4, 3, 1], &s);
    println!("\nTransfer function: {tf}");
    println!("DC gain: {}", tf.dc_gain());

    // Routh-Hurwitz stability criterion
    let coeffs = [symplex::int(1), symplex::int(3), symplex::int(4)];
    match is_routh_stable(&coeffs) {
        Some(true) => println!("Routh stable: yes"),
        Some(false) => println!("Routh stable: no"),
        None => println!("Routh stable: undetermined"),
    }

    println!("\n✓ Done!");
}
