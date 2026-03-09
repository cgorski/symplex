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


fn main() {
    println!("=== Laplace Transforms ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; t, s, n, z);

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
    let result = ctx.int(1).laplace(&t, &s);
    println!("L{{1}}       = {result}");

    // L{t} = 1/s²
    let result = t.laplace(&t, &s);
    println!("L{{t}}       = {result}");

    // L{t²} = 2/s³
    let result = t.powi(2).laplace(&t, &s);
    println!("L{{t²}}      = {result}");

    // L{exp(at)} = 1/(s-a)
    // L{exp(2t)} = 1/(s-2)
    let result = (&t * 2).exp().laplace(&t, &s);
    println!("L{{e^(2t)}}  = {result}");

    // L{exp(-3t)} = 1/(s+3)
    let result = (-&t * 3).exp().laplace(&t, &s);
    println!("L{{e^(-3t)}} = {result}");

    // L{sin(t)} = 1/(s²+1)
    let result = t.sin().laplace(&t, &s);
    println!("L{{sin(t)}}  = {result}");

    // L{cos(t)} = s/(s²+1)
    let result = t.cos().laplace(&t, &s);
    println!("L{{cos(t)}}  = {result}");

    // L{sin(3t)} = 3/(s²+9)
    let result = (&t * 3).sin().laplace(&t, &s);
    println!("L{{sin(3t)}} = {result}");

    // L{cos(3t)} = s/(s²+9)
    let result = (&t * 3).cos().laplace(&t, &s);
    println!("L{{cos(3t)}} = {result}");

    // ── Linearity: L{a·f + b·g} = a·L{f} + b·L{g} ────────────────

    println!("\n--- Linearity ---");

    // L{3·sin(t) + 2·cos(t)} should equal 3/(s²+1) + 2s/(s²+1)
    let combined = &(&t.sin() * 3) + &(&t.cos() * 2);
    let result = combined.laplace(&t, &s);
    println!("L{{3·sin(t) + 2·cos(t)}} = {result}");

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
    let result = f1.inverse_laplace(&s, &t);
    println!("L⁻¹{{1/s}}       = {result}");

    // L⁻¹{1/s²} = t
    let f2 = 1 / &s.powi(2);
    let result = f2.inverse_laplace(&s, &t);
    println!("L⁻¹{{1/s²}}      = {result}");

    // L⁻¹{1/(s-2)} = exp(2t)
    let f3 = 1 / &(&s - 2);
    let result = f3.inverse_laplace(&s, &t);
    println!("L⁻¹{{1/(s-2)}}   = {result}");

    // L⁻¹{1/(s+3)} = exp(-3t)
    let f4 = 1 / &(&s + 3);
    let result = f4.inverse_laplace(&s, &t);
    println!("L⁻¹{{1/(s+3)}}   = {result}");

    // L⁻¹{s/(s²+1)} = cos(t)
    let f5 = &s / &(expr!(ctx, s ^ 2 + 1));
    let result = f5.inverse_laplace(&s, &t);
    println!("L⁻¹{{s/(s²+1)}}  = {result}");

    // L⁻¹{1/(s²+1)} = sin(t)
    let f6 = 1 / &(expr!(ctx, s ^ 2 + 1));
    let result = f6.inverse_laplace(&s, &t);
    println!("L⁻¹{{1/(s²+1)}}  = {result}");

    // ── Roundtrip verification: L⁻¹{L{f}} = f ─────────────────────

    println!("\n--- Roundtrip Verification ---");

    let test_functions: Vec<(&str, Ex)> = vec![
        ("1", ctx.int(1)),
        ("t", t.clone()),
        ("exp(2t)", (&t * 2).exp()),
        ("sin(t)", t.sin()),
        ("cos(t)", t.cos()),
    ];

    for (name, f) in &test_functions {
        let fs = f.laplace(&t, &s);
        if fs.has_unevaluated() {
            println!("  L{{{name}}} — forward failed");
        } else {
            let roundtrip = fs.inverse_laplace(&s, &t);
            if roundtrip.has_unevaluated() {
                println!("  L⁻¹{{L{{{name}}}}} — inverse failed");
            } else {
                let simplified = roundtrip.simplify();
                println!("  L⁻¹{{L{{{name}}}}} = {simplified}");
            }
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
    let gs_expr = 1 / &(expr!(ctx, s ^ 2 + 3 * s + 4));
    println!("\n  G(s) as expression: {gs_expr}");
    let ht = gs_expr.inverse_laplace(&s, &t);
    println!("  Impulse response h(t) = {ht}");

    // Step response: L⁻¹{G(s)/s}
    let step_s = &gs_expr / &s;
    let step_t = step_s.inverse_laplace(&s, &t);
    println!("  Step response y(t) = {step_t}");

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
    match ctx.int(1).z_transform(&n, &z) {
        Ok(result) => println!("Z{{1}}       = {result}"),
        Err(e) => println!("Z{{1}}       failed: {e}"),
    }

    // Z{(1/2)^n} = z/(z - 1/2)
    let half = ctx.rational(1, 2);
    match half.pow(&n).z_transform(&n, &z) {
        Ok(result) => println!("Z{{(1/2)^n}} = {result}"),
        Err(e) => println!("Z{{(1/2)^n}} failed: {e}"),
    }

    // Z{2^n} = z/(z - 2)
    match ctx.int(2).pow(&n).z_transform(&n, &z) {
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
