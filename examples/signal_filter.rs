//! Digital Filter Design — discretize a continuous filter and generate code.
//!
//! This example demonstrates a real DSP engineering workflow:
//!
//!   1. Define a first-order low-pass filter in the s-domain
//!   2. Apply the bilinear (Tustin) transform analytically
//!   3. Derive the difference equation coefficients symbolically
//!   4. Simulate the step response to verify correctness
//!   5. Generate optimized Rust code for the filter update
//!
//! Run with: `cargo run --example signal_filter`

use symplex::prelude::*;

fn main() {
    println!("=== Digital Filter Design via Bilinear Transform ===\n");

    let ctx = Context::new();

    // ── 1. Continuous-time low-pass filter ──────────────────────────
    //
    // First-order low-pass: H(s) = ωc / (s + ωc)
    // With cutoff frequency ωc = 1 rad/s (normalized):
    //   H(s) = 1 / (s + 1)
    //
    // DC gain: H(0) = 1.  At cutoff: |H(j)| = 1/√2 ≈ -3 dB.

    println!("--- Continuous-Time Filter ---");
    println!("  H(s) = 1 / (s + 1)");
    println!("  Type: First-order low-pass, ωc = 1 rad/s");

    // ── 2. Bilinear transform ──────────────────────────────────────
    //
    // The bilinear (Tustin) transform maps s → (2/T)·(z-1)/(z+1).
    //
    // For T = 0.1 s (sampling rate = 10 Hz):
    //   s → 20·(z - 1) / (z + 1)
    //
    // Substituting into H(s) = 1/(s + 1):
    //
    //   H(z) = 1 / (20·(z-1)/(z+1) + 1)
    //        = (z + 1) / (20·(z - 1) + (z + 1))
    //        = (z + 1) / (20z - 20 + z + 1)
    //        = (z + 1) / (21z - 19)
    //
    // Dividing numerator and denominator by z:
    //   H(z) = (1 + z⁻¹) / (21 - 19·z⁻¹)
    //
    // Normalizing (divide by 21):
    //   H(z) = (1/21 + 1/21·z⁻¹) / (1 - 19/21·z⁻¹)
    //
    // So: b0 = 1/21, b1 = 1/21, a1 = -19/21

    println!("\n--- Bilinear Transform (T = 0.1 s, fs = 10 Hz) ---");
    println!("  s → 20·(z-1)/(z+1)");
    println!();
    println!("  H(z) = (z + 1) / (21z - 19)");
    println!("       = (1/21 + 1/21·z⁻¹) / (1 - 19/21·z⁻¹)");

    // Use symplex to verify the algebra symbolically.
    // Build: N(z) = z + 1 and D(z) = 21z - 19, check H(z=1) = DC gain.
    symplex::syms!(ctx; z);
    let numer = &z + 1;
    let denom = &z * 21 - 19;

    // DC gain: H(z=1) = H(s=0) = 1 (z=1 maps to s=0 in bilinear)
    let dc_gain = numer.subs(&z, &ctx.int(1)).eval_f64().unwrap()
        / denom.subs(&z, &ctx.int(1)).eval_f64().unwrap();
    println!("\n  DC gain check: H(z=1) = {dc_gain:.4} (should be 1.0)");
    assert!((dc_gain - 1.0).abs() < 1e-10, "DC gain must be 1");

    // Nyquist: H(z=-1) = H(s=∞) = 0 (z=-1 maps to s=∞)
    let nyquist = numer.subs(&z, &ctx.int(-1)).eval_f64().unwrap()
        / denom.subs(&z, &ctx.int(-1)).eval_f64().unwrap();
    println!("  Nyquist check: H(z=-1) = {nyquist:.4} (should be 0.0)");
    assert!(nyquist.abs() < 1e-10, "Nyquist gain must be 0");

    // ── 3. Difference equation coefficients ─────────────────────────
    //
    // H(z) = Y(z)/X(z) = (b0 + b1·z⁻¹) / (1 + a1·z⁻¹)
    //
    // Cross-multiply: Y(z)·(1 + a1·z⁻¹) = X(z)·(b0 + b1·z⁻¹)
    //
    // In the time domain:
    //   y[n] + a1·y[n-1] = b0·x[n] + b1·x[n-1]
    //   y[n] = b0·x[n] + b1·x[n-1] - a1·y[n-1]

    let b0: f64 = 1.0 / 21.0;
    let b1: f64 = 1.0 / 21.0;
    let a1: f64 = -19.0 / 21.0;

    println!("\n--- Difference Equation Coefficients ---");
    println!("  b0 =  1/21 ≈ {b0:.10}");
    println!("  b1 =  1/21 ≈ {b1:.10}");
    println!("  a1 = -19/21 ≈ {a1:.10}");
    println!();
    println!("  y[n] = b0·x[n] + b1·x[n-1] - a1·y[n-1]");
    println!("       = {b0:.4}·x[n] + {b1:.4}·x[n-1] + {:.4}·y[n-1]", -a1);

    // ── 4. Build symbolic filter and generate code ──────────────────
    //
    // Use exact rationals for the coefficients — symplex keeps
    // everything in Ratio<BigInt>, so 1/21 is exact, not 0.047619...

    println!("\n--- Symbolic Filter Expression ---");

    symplex::syms!(ctx; x_n, x_prev, y_prev);
    let b0_exact = ctx.rational(1, 21);
    let b1_exact = ctx.rational(1, 21);
    let neg_a1_exact = ctx.rational(19, 21);

    let filter_expr = &b0_exact * &x_n + &b1_exact * &x_prev + &neg_a1_exact * &y_prev;
    let filter_eval = filter_expr.eval();
    println!("  y[n] = {filter_eval}");

    // Generate Rust code
    println!("\n--- Generated Rust Code ---");
    match filter_eval.to_rust_fn("filter_update", &["x_n", "x_prev", "y_prev"]) {
        Ok(code) => println!("{code}"),
        Err(e) => {
            println!("  Codegen note: {e}");
            println!("  // Fallback implementation:");
            println!("  fn filter_update(x_n: f64, x_prev: f64, y_prev: f64) -> f64 {{");
            println!("      (1.0/21.0) * x_n + (1.0/21.0) * x_prev + (19.0/21.0) * y_prev");
            println!("  }}");
        }
    }

    // Compile to closure for simulation
    let filter_fn = filter_eval
        .compile(&["x_n", "x_prev", "y_prev"])
        .expect("compile filter");

    // Quick sanity check
    println!("\nCompiled filter verification:");
    println!(
        "  y = f(1.0, 0.0, 0.0) = {:.6}",
        filter_fn(&[1.0, 0.0, 0.0])
    );
    println!("  Expected: b0·1 = 1/21 ≈ {:.6}", 1.0 / 21.0);

    // ── 5. Step response simulation ─────────────────────────────────
    //
    // Apply a unit step (x[n] = 1 for all n ≥ 0) and watch the
    // output converge to the DC gain of 1.0.

    println!("\n--- Step Response (unit step input) ---");
    println!("    n | x[n] | y[n]     | error");
    println!("   ---|------|----------|--------");

    let mut y_p = 0.0_f64;
    let mut x_p = 0.0_f64;
    let n_samples = 100;

    for n in 0..n_samples {
        let x_curr = 1.0;
        let y_curr = filter_fn(&[x_curr, x_p, y_p]);
        let error = (y_curr - 1.0).abs();

        if n < 5 || n == n_samples - 1 {
            println!(
                "   {:>3} | {:.1}  | {:.6} | {:.6}",
                n, x_curr, y_curr, error
            );
        } else if n == 5 {
            println!("   ... |  ..  |    ..    |   ..");
        }

        x_p = x_curr;
        y_p = y_curr;
    }

    println!("\n  Final value: {y_p:.8}");
    println!("  Expected:    1.0 (DC gain)");
    assert!(
        (y_p - 1.0).abs() < 0.01,
        "step response should converge toward DC gain = 1"
    );

    // ── 6. Frequency response at a few points ──────────────────────
    //
    // |H(e^{jω})| at ω = 0, π/4, π/2, π

    println!("\n--- Frequency Response ---");
    let freqs = [
        (0.0, "DC (ω=0)"),
        (std::f64::consts::FRAC_PI_4, "ω=π/4"),
        (std::f64::consts::FRAC_PI_2, "ω=π/2"),
        (std::f64::consts::PI, "Nyquist (ω=π)"),
    ];

    for (omega, label) in &freqs {
        // z = e^{jω} = cos(ω) + j·sin(ω)
        let cos_w = omega.cos();
        let sin_w = omega.sin();

        // H(z) = (z + 1) / (21z - 19)
        // Numerator: (cos+1) + j·sin
        let nr = cos_w + 1.0;
        let ni = sin_w;
        // Denominator: (21·cos - 19) + j·21·sin
        let dr = 21.0 * cos_w - 19.0;
        let di = 21.0 * sin_w;

        let mag_n = (nr * nr + ni * ni).sqrt();
        let mag_d = (dr * dr + di * di).sqrt();
        let mag = mag_n / mag_d;
        let db = 20.0 * mag.log10();

        println!("  {:<16} |H| = {:.4}  ({:.1} dB)", label, mag, db);
    }

    // ── 7. Comparison with ideal continuous filter ──────────────────
    //
    // At ω_analog = ωc = 1 rad/s, the continuous filter has -3 dB.
    // The bilinear transform warps frequencies: ω_digital = 2/T · arctan(ω_analog · T/2).
    // For ωc=1, T=0.1: ω_digital = 20·arctan(0.05) ≈ 0.9992 rad/sample.

    let omega_digital = 20.0 * (0.05_f64).atan();
    println!("\n--- Frequency Warping ---");
    println!("  Analog cutoff:  ωc = 1.0 rad/s");
    println!("  Digital cutoff: ωd = 2/T·arctan(ωc·T/2) = {omega_digital:.4} rad/sample");
    println!("  Warping is minimal at low frequencies (T << 1/ωc)");

    println!("\n✓ Done!");
}
