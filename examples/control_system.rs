//! Control System Analysis — state-space models, stability, and design.
//!
//! Demonstrates:
//! - State-space construction from a physical system (mass-spring-damper)
//! - Pole analysis and stability checking
//! - Controllability and observability
//! - Transfer function representation
//! - Routh-Hurwitz stability criterion
//! - Ackermann pole placement
//! - Zero-order hold (ZOH) discretization
//! - Laplace transform usage for transfer function derivation
//!
//! Run with: cargo run --example control_system

use symplex::control::*;
use symplex::prelude::*;


fn main() {
    println!("=== Control System Analysis ===\n");

    // ════════════════════════════════════════════════════════════════
    // Part 1: Mass-Spring-Damper System
    // ════════════════════════════════════════════════════════════════

    println!("--- Mass-Spring-Damper State-Space Model ---\n");

    let __ctx = symplex::default_context();
    symplex::syms!(__ctx; s);

    // Physical system: mẍ + cẋ + kx = F
    // With m=1, c=3, k=4:
    //   ẍ + 3ẋ + 4x = F
    //
    // State variables: x1 = x (position), x2 = ẋ (velocity)
    // State equations:
    //   ẋ1 = x2
    //   ẋ2 = -4·x1 - 3·x2 + F
    //
    // Output: y = x1 (position measurement)

    let a = matrix![[0, 1], [-4, -3]];
    let b = matrix![[0], [1]];
    let c = matrix![[1, 0]];
    let d = matrix![[0]];

    let sys = StateSpace::new(a.clone(), b.clone(), c.clone(), d.clone());

    println!("State-space matrices:");
    println!("  A = {a}");
    println!("  B = {b}");
    println!("  C = {c}");
    println!("  D = {d}");
    println!(
        "\nDimensions: {} states, {} inputs, {} outputs",
        sys.num_states(),
        sys.num_inputs(),
        sys.num_outputs()
    );

    // ── Poles (eigenvalues of A) ───────────────────────────────────

    println!("\n--- Pole Analysis ---");
    let poles = sys.poles(&s);
    println!(
        "Poles: {:?}",
        poles.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
    );

    // Characteristic polynomial: det(sI - A) = s² + 3s + 4
    let char_p = sys.char_poly(&s);
    println!("Characteristic polynomial: {char_p}");

    // ── Stability ──────────────────────────────────────────────────

    println!("\n--- Stability ---");
    match sys.is_stable() {
        Some(true) => println!("Stable: yes (all poles have negative real part)"),
        Some(false) => println!("Stable: no"),
        None => println!("Stable: undetermined"),
    }

    // ── Controllability and Observability ───────────────────────────

    println!("\n--- Controllability & Observability ---");
    println!("Controllable: {}", sys.is_controllable());
    println!("Observable:   {}", sys.is_observable());

    let ctrb = sys.controllability_matrix();
    println!("Controllability matrix: {ctrb}");
    println!("  rank = {}", ctrb.rank());

    let obsv = sys.observability_matrix();
    println!("Observability matrix: {obsv}");
    println!("  rank = {}", obsv.rank());

    // ════════════════════════════════════════════════════════════════
    // Part 2: Transfer Function
    // ════════════════════════════════════════════════════════════════

    println!("\n\n--- Transfer Function ---\n");

    // G(s) = C(sI - A)⁻¹B + D = 1 / (s² + 3s + 4)
    let tf = TransferFunction::from_coeffs(&[1], &[4, 3, 1], &s);
    println!("Transfer function: {tf}");

    // DC gain: G(0) = 1/4
    println!("DC gain: {}", tf.dc_gain());

    // Poles and zeros of the transfer function
    let tf_poles = tf.poles();
    println!(
        "TF poles: {:?}",
        tf_poles.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
    );

    let tf_zeros = tf.zeros();
    println!(
        "TF zeros: {:?}",
        tf_zeros.iter().map(|z| format!("{z}")).collect::<Vec<_>>()
    );

    // ── Transfer function algebra ──────────────────────────────────

    println!("\n--- Transfer Function Algebra ---");

    // Series connection: G1(s) · G2(s)
    let g1 = TransferFunction::from_coeffs(&[1], &[1, 1], &s); // 1/(s+1)
    let g2 = TransferFunction::from_coeffs(&[1], &[2, 1], &s); // 1/(s+2)
    let series = g1.series(&g2);
    println!("G1 = {g1}");
    println!("G2 = {g2}");
    println!("G1·G2 (series) = {series}");

    // Parallel connection: G1(s) + G2(s)
    let parallel = g1.parallel(&g2);
    println!("G1+G2 (parallel) = {parallel}");

    // Unity feedback: G1/(1 + G1)
    let feedback = g1.feedback();
    println!("G1/(1+G1) (unity feedback) = {feedback}");

    // Feedback with sensor: G1/(1 + G1·G2)
    let feedback_with = g1.feedback_with(&g2);
    println!("G1/(1+G1·G2) = {feedback_with}");

    // ════════════════════════════════════════════════════════════════
    // Part 3: Routh-Hurwitz Stability Criterion
    // ════════════════════════════════════════════════════════════════

    println!("\n\n--- Routh-Hurwitz Stability ---\n");

    // Characteristic polynomial: s² + 3s + 4
    // All coefficients positive → necessary condition met
    let coeffs = [symplex::default_context().int(1), symplex::default_context().int(3), symplex::default_context().int(4)];
    match is_routh_stable(&coeffs) {
        Some(true) => println!("s² + 3s + 4: Routh stable (yes)"),
        Some(false) => println!("s² + 3s + 4: Routh stable (no)"),
        None => println!("s² + 3s + 4: Routh stable (undetermined)"),
    }

    // Routh array for a more interesting polynomial: s³ + 2s² + 3s + 4
    let coeffs3 = [
        symplex::default_context().int(1),
        symplex::default_context().int(2),
        symplex::default_context().int(3),
        symplex::default_context().int(4),
    ];
    match is_routh_stable(&coeffs3) {
        Some(true) => println!("s³ + 2s² + 3s + 4: Routh stable (yes)"),
        Some(false) => println!("s³ + 2s² + 3s + 4: Routh stable (no)"),
        None => println!("s³ + 2s² + 3s + 4: Routh stable (undetermined)"),
    }

    // Print the Routh array
    let routh = routh_array(&coeffs3);
    println!("\nRouth array for s³ + 2s² + 3s + 4:");
    for (i, row) in routh.iter().enumerate() {
        let entries: Vec<String> = row.iter().map(|e| format!("{e}")).collect();
        println!("  Row {i}: [{}]", entries.join(", "));
    }

    // Unstable example: s³ + s² - 2s + 1
    let unstable_coeffs = [
        symplex::default_context().int(1),
        symplex::default_context().int(1),
        symplex::default_context().int(-2),
        symplex::default_context().int(1),
    ];
    match is_routh_stable(&unstable_coeffs) {
        Some(true) => println!("\ns³ + s² - 2s + 1: Routh stable (yes)"),
        Some(false) => println!("\ns³ + s² - 2s + 1: Routh stable (no — sign change in first column)"),
        None => println!("\ns³ + s² - 2s + 1: Routh stable (undetermined)"),
    }

    // ════════════════════════════════════════════════════════════════
    // Part 4: Ackermann Pole Placement
    // ════════════════════════════════════════════════════════════════

    println!("\n\n--- Ackermann Pole Placement ---\n");

    // Place poles at s = -5 and s = -6
    // (faster response than the original poles at ≈ -1.5 ± j1.32)
    let desired_poles = [symplex::default_context().int(-5), symplex::default_context().int(-6)];

    println!(
        "Desired poles: {:?}",
        desired_poles
            .iter()
            .map(|p| format!("{p}"))
            .collect::<Vec<_>>()
    );

    match sys.ackermann(&desired_poles) {
        Some(k) => {
            println!("Feedback gain K = {k}");

            // Verify: eigenvalues of (A - BK) should be the desired poles
            let bk = &b * &k;
            let a_cl = &a - &bk;
            let cl_poles = a_cl.eigenvals(&s).unwrap();
            println!(
                "Closed-loop poles: {:?}",
                cl_poles
                    .iter()
                    .map(|p| format!("{p}"))
                    .collect::<Vec<_>>()
            );
        }
        None => {
            println!("Ackermann failed (system not controllable or singular)");
        }
    }

    // Place poles at s = -2 ± 3j (complex conjugate pair)
    // Note: we express these symbolically
    let i_unit = symplex::default_context().i_unit();
    let p1 = &symplex::default_context().int(-2) + &(&i_unit * 3);
    let p2 = &symplex::default_context().int(-2) - &(&i_unit * 3);
    println!("\nDesired poles: {p1}, {p2}");

    match sys.ackermann(&[p1, p2]) {
        Some(k) => {
            println!("Feedback gain K = {k}");
        }
        None => {
            println!("Ackermann failed for complex poles");
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Part 5: ZOH Discretization
    // ════════════════════════════════════════════════════════════════

    println!("\n\n--- ZOH Discretization ---\n");

    // Discretize with sample time dt = 0.01s
    // Uses Taylor series approximation of the matrix exponential
    let dt = symplex::default_context().rational(1, 100); // 0.01 s
    println!("Sample time: dt = {dt} s");

    let discrete = sys.discretize_zoh(&dt, 4);
    println!("\nDiscrete-time system (4th-order Taylor approx):");
    println!(
        "  States: {}, Inputs: {}, Outputs: {}",
        discrete.num_states(),
        discrete.num_inputs(),
        discrete.num_outputs()
    );

    // Evaluate the discrete A matrix numerically
    println!("\nDiscrete A matrix (Ad):");
    let ad = &discrete.a;
    println!("  {ad}");

    println!("\nDiscrete B matrix (Bd):");
    let bd = &discrete.b;
    println!("  {bd}");

    // Check discrete-time stability: all eigenvalues inside unit circle
    let disc_poles = discrete.poles(&s);
    println!(
        "\nDiscrete poles: {:?}",
        disc_poles
            .iter()
            .map(|p| format!("{p}"))
            .collect::<Vec<_>>()
    );

    match discrete.is_stable() {
        Some(true) => println!("Discrete system stable: yes"),
        Some(false) => println!("Discrete system stable: no"),
        None => println!("Discrete system stable: undetermined"),
    }

    // ════════════════════════════════════════════════════════════════
    // Part 6: Laplace Transform Usage
    // ════════════════════════════════════════════════════════════════

    println!("\n\n--- Laplace Transform ---\n");

    let __ctx = symplex::default_context();
    symplex::syms!(__ctx; t);

    // Derive transfer function from impulse response
    // For the mass-spring-damper, the impulse response is the
    // inverse Laplace of G(s) = 1/(s² + 3s + 4)
    let gs = 1 / &(expr!(s ^ 2 + 3 * s + 4));
    println!("G(s) = {gs}");

    // Inverse Laplace to get impulse response h(t)
    let ht = gs.inverse_laplace(&s, &t);
    println!("h(t) = L⁻¹{{G(s)}} = {ht}");

    // Forward Laplace of some common signals
    println!("\nCommon Laplace pairs:");

    // L{1} = 1/s
    let result = symplex::default_context().int(1).laplace(&t, &s);
    println!("  L{{1}} = {result}");

    // L{t} = 1/s²
    let result = t.laplace(&t, &s);
    println!("  L{{t}} = {result}");

    // L{exp(-3t)} = 1/(s+3)
    let result = (-&t * 3).exp().laplace(&t, &s);
    println!("  L{{exp(-3t)}} = {result}");

    // L{sin(2t)} = 2/(s²+4)
    let result = (&t * 2).sin().laplace(&t, &s);
    println!("  L{{sin(2t)}} = {result}");

    // L{cos(2t)} = s/(s²+4)
    let result = (&t * 2).cos().laplace(&t, &s);
    println!("  L{{cos(2t)}} = {result}");

    // ── Evaluate transfer function at specific frequency ───────────
    println!("\n--- Frequency Response ---");
    let omega_vals = [1i64, 2, 5, 10];
    for omega in &omega_vals {
        // G(jω) — evaluate at s = jω
        let jw = &i_unit * *omega;
        let g_jw = gs.subs(&s, &jw).eval();
        println!("  G(j·{omega}) = {g_jw}");
    }

    println!("\n✓ Done!");
}
