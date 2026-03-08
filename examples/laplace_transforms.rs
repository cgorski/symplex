//! Laplace Transforms — forward, inverse, and transfer function derivation.
//!
//! Demonstrates:
//! - Forward Laplace transform of common signals
//! - Inverse Laplace transform via partial fractions
//! - Transfer function derivation from differential equations
//! - Z-transform for discrete-time systems
//! - Linearity and shifting properties
//!
//! Run with: cargo run --example laplace_transforms

use symplex::prelude::*;
use symplex::vars;

fn main() {
    println!("=== Laplace Transforms ===\n");

    vars!(t, s, n, z);

    // ════════════════════════════════════════════════════════════════
    // Part 1: Forward Laplace Transform
    // ════════════════════════════════════════════════════════════════
    //
    // The Laplace transform converts a time-domain function f(t)
    // into a frequency-domain function F(s):
    //
    //   L{f(t)} = F(s) = ∫₀^∞ f(t)·e^(-st) dt
    //
    // Symplex uses a table of known transforms plus linearity.

    println!("--- Forward Laplace Transforms ---\n");

    // L{1} = 1/s
    match symplex::default_context().int(1).laplace(&t, &s) {
        Ok(result) => println!("L{{1}}       = {result}"),
        Err(e) => println!("L{{1}}       failed: {e}"),
    }

    // L{t} = 1/s²
    match t.laplace(&t, &s) {
        Ok(result) => println!("L{{t}}       = {result}"),
        Err(e) => println!("L{{t}}       failed: {e}"),
    }

    // L{t²} = 2/s³
    match t.powi(2).laplace(&t, &s) {
        Ok(result) => println!("L{{t²}}      = {result}"),
        Err(e) => println!("L{{t²}}      failed: {e}"),
    }

    // L{exp(at)} = 1/(s-a)
    // L{exp(2t)} = 1/(s-2)
    match (&t * 2).exp().laplace(&t, &s) {
        Ok(result) => println!("L{{e^(2t)}}  = {result}"),
        Err(e) => println!("L{{e^(2t)}}  failed: {e}"),
    }

    // L{exp(-3t)} = 1/(s+3)
    match (-&t * 3).exp().laplace(&t, &s) {
        Ok(result) => println!("L{{e^(-3t)}} = {result}"),
        Err(e) => println!("L{{e^(-3t)}} failed: {e}"),
    }

    // L{sin(t)} = 1/(s²+1)
    match t.sin().laplace(&t, &s) {
        Ok(result) => println!("L{{sin(t)}}  = {result}"),
        Err(e) => println!("L{{sin(t)}}  failed: {e}"),
    }

    // L{cos(t)} = s/(s²+1)
    match t.cos().laplace(&t, &s) {
        Ok(result) => println!("L{{cos(t)}}  = {result}"),
        Err(e) => println!("L{{cos(t)}}  failed: {e}"),
    }

    // L{sin(3t)} = 3/(s²+9)
    match (&t * 3).sin().laplace(&t, &s) {
        Ok(result) => println!("L{{sin(3t)}} = {result}"),
        Err(e) => println!("L{{sin(3t)}} failed: {e}"),
    }

    // L{cos(3t)} = s/(s²+9)
    match (&t * 3).cos().laplace(&t, &s) {
        Ok(result) => println!("L{{cos(3t)}} = {result}"),
        Err(e) => println!("L{{cos(3t)}} failed: {e}"),
    }

    // ── Linearity: L{a·f + b·g} = a·L{f} + b·L{g} ────────────────

    println!("\n--- Linearity ---");

    // L{3·sin(t) + 2·cos(t)} should equal 3/(s²+1) + 2s/(s²+1)
    let combined = &(&t.sin() * 3) + &(&t.cos() * 2);
    match combined.laplace(&t, &s) {
        Ok(result) => println!("L{{3·sin(t) + 2·cos(t)}} = {result}"),
        Err(e) => println!("L{{3·sin(t) + 2·cos(t)}} failed: {e}"),
    }

    // ════════════════════════════════════════════════════════════════
    // Part 2: Inverse Laplace Transform
    // ════════════════════════════════════════════════════════════════
    //
    // The inverse Laplace transform recovers the time-domain function
    // from F(s).  Symplex uses partial fraction decomposition followed
    // by table lookup for each term.

    println!("\n\n--- Inverse Laplace Transforms ---\n");

    // L⁻¹{1/s} = 1
    let f1 = 1 / &s;
    match f1.inverse_laplace(&s, &t) {
        Ok(result) => println!("L⁻¹{{1/s}}       = {result}"),
        Err(e) => println!("L⁻¹{{1/s}}       failed: {e}"),
    }

    // L⁻¹{1/s²} = t
    let f2 = 1 / &s.powi(2);
    match f2.inverse_laplace(&s, &t) {
        Ok(result) => println!("L⁻¹{{1/s²}}      = {result}"),
        Err(e) => println!("L⁻¹{{1/s²}}      failed: {e}"),
    }

    // L⁻¹{1/(s-2)} = exp(2t)
    let f3 = 1 / &(&s - 2);
    match f3.inverse_laplace(&s, &t) {
        Ok(result) => println!("L⁻¹{{1/(s-2)}}   = {result}"),
        Err(e) => println!("L⁻¹{{1/(s-2)}}   failed: {e}"),
    }

    // L⁻¹{1/(s+3)} = exp(-3t)
    let f4 = 1 / &(&s + 3);
    match f4.inverse_laplace(&s, &t) {
        Ok(result) => println!("L⁻¹{{1/(s+3)}}   = {result}"),
        Err(e) => println!("L⁻¹{{1/(s+3)}}   failed: {e}"),
    }

    // L⁻¹{s/(s²+1)} = cos(t)
    let f5 = &s / &(expr!(s ^ 2 + 1));
    match f5.inverse_laplace(&s, &t) {
        Ok(result) => println!("L⁻¹{{s/(s²+1)}}  = {result}"),
        Err(e) => println!("L⁻¹{{s/(s²+1)}}  failed: {e}"),
    }

    // L⁻¹{1/(s²+1)} = sin(t)
    let f6 = 1 / &(expr!(s ^ 2 + 1));
    match f6.inverse_laplace(&s, &t) {
        Ok(result) => println!("L⁻¹{{1/(s²+1)}}  = {result}"),
        Err(e) => println!("L⁻¹{{1/(s²+1)}}  failed: {e}"),
    }

    // ── Roundtrip verification: L⁻¹{L{f}} = f ─────────────────────

    println!("\n--- Roundtrip Verification ---");

    let test_functions: Vec<(&str, Ex)> = vec![
        ("1", symplex::default_context().int(1)),
        ("t", t.clone()),
        ("exp(2t)", (&t * 2).exp()),
        ("sin(t)", t.sin()),
        ("cos(t)", t.cos()),
    ];

    for (name, f) in &test_functions {
        match f.laplace(&t, &s) {
            Ok(fs) => match fs.inverse_laplace(&s, &t) {
                Ok(roundtrip) => {
                    let simplified = roundtrip.simplify();
                    println!("  L⁻¹{{L{{{name}}}}} = {simplified}");
                }
                Err(_) => println!("  L⁻¹{{L{{{name}}}}} — inverse failed"),
            },
            Err(_) => println!("  L{{{name}}} — forward failed"),
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Part 3: Transfer Function Derivation
    // ════════════════════════════════════════════════════════════════
    //
    // For a mass-spring-damper: mẍ + cẋ + kx = F(t)
    // Taking the Laplace transform (zero initial conditions):
    //   (ms² + cs + k)X(s) = F(s)
    //   G(s) = X(s)/F(s) = 1/(ms² + cs + k)

    println!("\n\n--- Transfer Function Derivation ---\n");

    // G(s) = 1/(s² + 3s + 4) for m=1, c=3, k=4
    let tf = TransferFunction::from_coeffs(&[1], &[4, 3, 1], &s);
    println!("Mass-spring-damper transfer function:");
    println!("  G(s) = {tf}");
    println!("  DC gain G(0) = {}", tf.dc_gain());

    // Poles of the transfer function
    let poles = tf.poles();
    println!(
        "  Poles: {:?}",
        poles.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
    );

    // Zeros of the transfer function
    let zeros = tf.zeros();
    println!(
        "  Zeros: {:?}",
        zeros.iter().map(|z| format!("{z}")).collect::<Vec<_>>()
    );

    // Impulse response: h(t) = L⁻¹{G(s)}
    let gs_expr = 1 / &(expr!(s ^ 2 + 3 * s + 4));
    println!("\n  G(s) as expression: {gs_expr}");
    match gs_expr.inverse_laplace(&s, &t) {
        Ok(ht) => println!("  Impulse response h(t) = {ht}"),
        Err(e) => println!("  Impulse response failed: {e}"),
    }

    // Step response: L⁻¹{G(s)/s}
    let step_s = &gs_expr / &s;
    match step_s.inverse_laplace(&s, &t) {
        Ok(step_t) => println!("  Step response y(t) = {step_t}"),
        Err(e) => println!("  Step response failed: {e}"),
    }

    // ── Series and feedback connections ─────────────────────────────

    println!("\n--- Transfer Function Algebra ---");

    let g1 = TransferFunction::from_coeffs(&[1], &[1, 1], &s); // 1/(s+1)
    let g2 = TransferFunction::from_coeffs(&[1], &[2, 1], &s); // 1/(s+2)

    println!("G1(s) = {g1}");
    println!("G2(s) = {g2}");

    let series_tf = g1.series(&g2);
    println!("Series:   G1·G2 = {series_tf}");

    let parallel_tf = g1.parallel(&g2);
    println!("Parallel: G1+G2 = {parallel_tf}");

    let feedback_tf = g1.feedback();
    println!("Unity feedback: G1/(1+G1) = {feedback_tf}");

    // ════════════════════════════════════════════════════════════════
    // Part 4: Z-Transform (Discrete-Time)
    // ════════════════════════════════════════════════════════════════
    //
    // The Z-transform is the discrete-time analog of the Laplace
    // transform, converting sequences x[n] into X(z).
    //
    //   Z{x[n]} = X(z) = Σ x[n]·z^(-n)

    println!("\n\n--- Z-Transform ---\n");

    // Z{1} (unit step) — should be z/(z-1)
    match symplex::default_context().int(1).z_transform(&n, &z) {
        Ok(result) => println!("Z{{1}}       = {result}"),
        Err(e) => println!("Z{{1}}       failed: {e}"),
    }

    // Z{(1/2)^n} = z/(z - 1/2)
    let half = symplex::default_context().rational(1, 2);
    match half.pow(&n).z_transform(&n, &z) {
        Ok(result) => println!("Z{{(1/2)^n}} = {result}"),
        Err(e) => println!("Z{{(1/2)^n}} failed: {e}"),
    }

    // Z{2^n} = z/(z - 2)
    match symplex::default_context().int(2).pow(&n).z_transform(&n, &z) {
        Ok(result) => println!("Z{{2^n}}     = {result}"),
        Err(e) => println!("Z{{2^n}}     failed: {e}"),
    }

    // ── Inverse Z-transform ────────────────────────────────────────

    println!("\n--- Inverse Z-Transform ---");

    // Z⁻¹{z/(z-2)} = 2^n
    let xz1 = &z / &(&z - 2);
    match xz1.inverse_z_transform(&z, &n) {
        Ok(result) => println!("Z⁻¹{{z/(z-2)}}   = {result}"),
        Err(e) => println!("Z⁻¹{{z/(z-2)}}   failed: {e}"),
    }

    // Z⁻¹{z/(z-1/2)} = (1/2)^n
    let xz2 = &z / &(&z - &half);
    match xz2.inverse_z_transform(&z, &n) {
        Ok(result) => println!("Z⁻¹{{z/(z-1/2)}} = {result}"),
        Err(e) => println!("Z⁻¹{{z/(z-1/2)}} failed: {e}"),
    }

    // ── Z-transform roundtrip ──────────────────────────────────────

    println!("\n--- Z-Transform Roundtrip ---");

    // Z⁻¹{Z{(1/2)^n}} should give back (1/2)^n
    match half.pow(&n).z_transform(&n, &z) {
        Ok(xz) => match xz.inverse_z_transform(&z, &n) {
            Ok(roundtrip) => {
                let simplified = roundtrip.simplify();
                println!("Z⁻¹{{Z{{(1/2)^n}}}} = {simplified}");
            }
            Err(e) => println!("Inverse failed: {e}"),
        },
        Err(e) => println!("Forward failed: {e}"),
    }

    // ════════════════════════════════════════════════════════════════
    // Summary
    // ════════════════════════════════════════════════════════════════

    println!("\n--- Summary of Transform Pairs ---\n");
    println!("  Laplace (continuous-time):");
    println!("    f(t)        ↔  F(s)");
    println!("    1           ↔  1/s");
    println!("    t           ↔  1/s²");
    println!("    exp(at)     ↔  1/(s-a)");
    println!("    sin(ωt)     ↔  ω/(s²+ω²)");
    println!("    cos(ωt)     ↔  s/(s²+ω²)");
    println!();
    println!("  Z-transform (discrete-time):");
    println!("    x[n]        ↔  X(z)");
    println!("    1           ↔  z/(z-1)");
    println!("    aⁿ          ↔  z/(z-a)");

    println!("\n✓ Done!");
}
