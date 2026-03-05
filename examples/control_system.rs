//! Control System Analysis — state-space models and stability.
//!
//! Analyzes a mass-spring-damper system using state-space representation,
//! checks stability, and discretizes for digital control.
//!
//! Run with: cargo run --example control_system

use symplex::control::*;
use symplex::matrix::Matrix;
use symplex::vars;

fn main() {
    println!("=== Symplex Control System Analysis ===\n");

    // Mass-spring-damper: mẍ + cẋ + kx = F
    // State-space form: ẋ = Ax + Bu, y = Cx
    // x₁ = position, x₂ = velocity
    // A = [[0, 1], [-k/m, -c/m]], B = [[0], [1/m]], C = [[1, 0]]

    vars!(s);

    // System parameters (k/m=4, c/m=3)
    let a = Matrix::new(vec![
        vec![symplex::int(0), symplex::int(1)],
        vec![symplex::int(-4), symplex::int(-3)],
    ]);
    let b = Matrix::new(vec![
        vec![symplex::int(0)],
        vec![symplex::int(1)],
    ]);
    let c = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(0)],
    ]);
    let d = Matrix::new(vec![
        vec![symplex::int(0)],
    ]);

    let sys = StateSpace::new(a, b, c, d);
    println!("System: ẋ = Ax + Bu, y = Cx");
    println!(
        "  States: {}, Inputs: {}, Outputs: {}",
        sys.num_states(),
        sys.num_inputs(),
        sys.num_outputs()
    );

    // Characteristic polynomial
    let cp = sys.char_poly(&s);
    println!("\nCharacteristic polynomial: {cp}");

    // Poles (eigenvalues)
    let poles = sys.poles(&s);
    println!(
        "Poles: {:?}",
        poles.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
    );

    // Stability
    match sys.is_stable() {
        Some(true) => println!("System is STABLE (all poles in LHP)"),
        Some(false) => println!("System is UNSTABLE"),
        None => println!("Stability could not be determined symbolically"),
    }

    // Controllability & Observability
    println!("\nControllable: {}", sys.is_controllable());
    println!("Observable: {}", sys.is_observable());

    // Transfer function: G(s) = 1 / (s² + 3s + 4)
    let tf = TransferFunction::new(
        symplex::int(1),
        &s.powi(2) + &(&s * 3) + 4,
        s.clone(),
    );
    println!("\nTransfer function: {tf}");
    println!("DC gain: {}", tf.dc_gain());

    // Discretize for digital control (dt = 0.01s)
    let dt = symplex::rational(1, 100);
    let discrete = sys.discretize_zoh(&dt, 6);
    println!("\nDiscretized (ZOH, dt=0.01s):");
    println!(
        "  Ad dimensions: {}x{}",
        discrete.a.nrows(),
        discrete.a.ncols()
    );

    // Routh-Hurwitz stability criterion
    // Characteristic polynomial: s² + 3s + 4, coefficients [1, 3, 4]
    let coeffs = [symplex::int(1), symplex::int(3), symplex::int(4)];
    match is_routh_stable(&coeffs) {
        Some(true) => println!("\nRouth-Hurwitz: STABLE"),
        Some(false) => println!("\nRouth-Hurwitz: UNSTABLE"),
        None => println!("\nRouth-Hurwitz: indeterminate"),
    }

    println!("\n✓ Analysis complete!");
}
