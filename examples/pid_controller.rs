//! PID Controller Design — model a plant, tune gains, verify stability, generate code.
//!
//! This example walks through a real controls engineering workflow:
//!
//!   1. Model a DC motor as a second-order transfer function
//!   2. Design a PID controller with symbolic gains Kp, Ki, Kd
//!   3. Compute the closed-loop characteristic polynomial
//!   4. Analyze stability via the Routh-Hurwitz criterion
//!   5. Choose gains to place poles, verify stability
//!   6. Generate optimized Rust code for the controller
//!
//! Run with: `cargo run --example pid_controller`

use symplex::prelude::*;
use symplex::control::*;

fn main() {
    println!("=== PID Controller Design for a DC Motor ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; s, t);

    // ── 1. Plant model ─────────────────────────────────────────────
    //
    // DC motor: G(s) = K / (s(Js + b))
    //   K = motor torque constant
    //   J = moment of inertia
    //   b = damping coefficient
    //
    // With J=1, b=10, K=20 (normalized):
    //   G(s) = 20 / (s² + 10s)

    println!("--- Plant Model ---");
    let plant_num = ctx.int(20);
    let plant_den = expr!(ctx, s ^ 2 + 10 * s);
    println!("G(s) = 20 / (s² + 10s)");
    println!("  Open-loop poles: s = 0, s = -10");

    // ── 2. PID controller ──────────────────────────────────────────
    //
    // C(s) = Kp + Ki/s + Kd·s = (Kd·s² + Kp·s + Ki) / s

    println!("\n--- PID Controller ---");
    symplex::syms!(ctx; Kp, Ki, Kd);
    let pid_num = expr!(ctx, Kd * s ^ 2 + Kp * s + Ki);
    let pid_den = s.clone();
    println!("C(s) = Kp + Ki/s + Kd·s");
    println!("     = (Kd·s² + Kp·s + Ki) / s");

    // ── 3. Closed-loop transfer function ───────────────────────────
    //
    // T(s) = C(s)G(s) / (1 + C(s)G(s))
    //
    // Numerator of C·G = 20(Kd·s² + Kp·s + Ki)
    // Denominator of C·G = s·(s² + 10s) = s³ + 10s²
    //
    // Closed-loop char. poly = den(C·G) + num(C·G)
    //   = s³ + 10s² + 20·Kd·s² + 20·Kp·s + 20·Ki
    //   = s³ + (10 + 20·Kd)s² + 20·Kp·s + 20·Ki

    println!("\n--- Closed-Loop Characteristic Polynomial ---");
    let char_poly = expr!(ctx, s ^ 3 + (10 + 20 * Kd) * s ^ 2 + 20 * Kp * s + 20 * Ki);
    println!("P(s) = {char_poly}");

    // ── 4. Routh-Hurwitz stability analysis ────────────────────────
    //
    // For s³ + a₂s² + a₁s + a₀, Routh conditions:
    //   a₂ > 0:  10 + 20·Kd > 0
    //   a₀ > 0:  20·Ki > 0
    //   a₂·a₁ > a₀:  (10 + 20·Kd)·(20·Kp) > 20·Ki

    println!("\n--- Routh-Hurwitz Stability Conditions ---");
    let a2 = expr!(ctx, 10 + 20 * Kd);
    let a1 = expr!(ctx, 20 * Kp);
    let a0 = expr!(ctx, 20 * Ki);
    println!("  a₂ = {a2} > 0");
    println!("  a₀ = {a0} > 0");
    println!("  a₂·a₁ > a₀: ({a2})·({a1}) > {a0}");
    let routh_product = (&a2 * &a1).expand();
    println!("  i.e., {routh_product} > {a0}");

    // ── 5. Choose gains and verify ─────────────────────────────────
    //
    // Pick Kp = 5, Ki = 2, Kd = 0.5
    //   a₂ = 10 + 10 = 20  > 0  ✓
    //   a₀ = 40              > 0  ✓
    //   a₂·a₁ = 20·100 = 2000  > 40  ✓

    println!("\n--- Gain Selection: Kp=5, Ki=2, Kd=0.5 ---");

    let gains = [(&Kp, 5i64), (&Ki, 2i64), (&Kd, 1i64)]; // Kd = 1 for now
    let char_concrete = char_poly
        .subs(&Kp, &ctx.int(5))
        .subs(&Ki, &ctx.int(2))
        .subs(&Kd, &ctx.rational(1, 2));
    let char_expanded = char_concrete.expand().eval();
    println!("P(s) = {char_expanded}");

    // Find the closed-loop poles
    let poles = char_expanded.solve(&s);
    println!("\nClosed-loop poles:");
    match &poles {
        Ok(roots) => {
            let mut all_stable = true;
            for (i, root) in roots.iter().enumerate() {
                let val = root.eval_f64();
                let stable = match &val {
                    Ok(v) => *v < 0.0,
                    Err(_) => {
                        // Complex root — check real part via eval_complex64
                        root.eval_complex64()
                            .map(|(re, _im)| re < 0.0)
                            .unwrap_or(false)
                    }
                };
                if !stable {
                    all_stable = false;
                }
                let val_str = val
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|_| format!("{root}"));
                println!("  p{} = {} {}", i + 1, val_str, if stable { "✓" } else { "✗" });
            }
            println!(
                "\nStability: {}",
                if all_stable {
                    "STABLE — all poles in left half-plane"
                } else {
                    "UNSTABLE — pole(s) in right half-plane"
                }
            );
        }
        Err(e) => println!("  Could not find poles analytically: {e}"),
    }

    // Verify Routh conditions numerically
    println!("\nRouth-Hurwitz verification:");
    let a2_val = a2.subs(&Kd, &ctx.rational(1, 2)).eval_f64().unwrap();
    let a1_val = a1.subs(&Kp, &ctx.int(5)).eval_f64().unwrap();
    let a0_val = a0.subs(&Ki, &ctx.int(2)).eval_f64().unwrap();
    println!("  a₂ = {a2_val} > 0: {}", a2_val > 0.0);
    println!("  a₀ = {a0_val} > 0: {}", a0_val > 0.0);
    println!(
        "  a₂·a₁ = {} > a₀ = {}: {}",
        a2_val * a1_val,
        a0_val,
        a2_val * a1_val > a0_val
    );

    // ── 6. Generate controller code ────────────────────────────────
    //
    // PID output: u(t) = Kp·e + Ki·∫e dt + Kd·de/dt
    // Discrete approximation (for embedded):
    //   u[k] = Kp·e[k] + Ki·Ts·Σe + Kd·(e[k] - e[k-1])/Ts
    //
    // Generate the update equation as Rust code.

    println!("\n--- Generated Controller Code ---");

    // Build the PID output expression with concrete gains
    symplex::syms!(ctx; error, integral, derivative);
    let kp_val = ctx.rational(5, 1);
    let ki_val = ctx.rational(2, 1);
    let kd_val = ctx.rational(1, 2);
    let pid_output = &kp_val * &error + &ki_val * &integral + &kd_val * &derivative;
    let pid_simplified = pid_output.eval();

    match pid_simplified.to_rust_fn("pid_update", &["error", "integral", "derivative"]) {
        Ok(code) => {
            println!("{code}");
        }
        Err(e) => {
            println!("  Code generation error: {e}");
            println!("  Expression: {pid_simplified}");
            // Fall back to manual code
            println!("\n  // Manual implementation:");
            println!("  fn pid_update(error: f64, integral: f64, derivative: f64) -> f64 {{");
            println!("      5.0 * error + 2.0 * integral + 0.5 * derivative");
            println!("  }}");
        }
    }

    // Compile and test numerically
    if let Some(f) = pid_simplified.compile(&["error", "integral", "derivative"]) {
        println!("Numerical verification:");
        // Step response: error=1.0, no integral or derivative yet
        println!("  u(e=1.0, i=0, d=0)   = {:.2}", f(&[1.0, 0.0, 0.0]));
        // With integral buildup
        println!("  u(e=0.5, i=2.0, d=-1) = {:.2}", f(&[0.5, 2.0, -1.0]));
        // Steady state: error=0
        println!("  u(e=0, i=3.0, d=0)    = {:.2}", f(&[0.0, 3.0, 0.0]));
    }

    println!("\n✓ Done!");
}
