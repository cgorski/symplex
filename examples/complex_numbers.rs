//! Complex Numbers — Euler's formula, complex roots, and evaluation.
//!
//! Demonstrates:
//! - The imaginary unit I and basic complex arithmetic
//! - Euler's formula: exp(I·x) = cos(x) + I·sin(x)
//! - Complex quadratic roots (x² + 1 = 0)
//! - Trig–hyperbolic bridge via complex rewriting
//! - Numerical complex evaluation with eval_complex64()
//! - Real and imaginary part extraction
//!
//! Run with: cargo run --example complex_numbers

use symplex::prelude::*;

fn main() {
    println!("=== Complex Numbers ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; x);

    // ── 1. The imaginary unit ──────────────────────────────────────
    println!("--- The Imaginary Unit ---");

    let i = ctx.i_unit();
    println!("I = {i}");

    // I² = -1
    let i_squared = i.powi(2).eval();
    println!("I² = {i_squared}");

    // I³ = -I
    let i_cubed = i.powi(3).eval();
    println!("I³ = {i_cubed}");

    // I⁴ = 1
    let i_fourth = i.powi(4).eval();
    println!("I⁴ = {i_fourth}");

    // ── 2. Complex arithmetic ──────────────────────────────────────
    println!("\n--- Complex Arithmetic ---");

    // (2 + 3I)
    let z1 = &ctx.int(2) + &(&i * 3);
    println!("z₁ = {z1}");

    // (1 - 2I)
    let z2 = &ctx.int(1) - &(&i * 2);
    println!("z₂ = {z2}");

    // z1 + z2
    let sum = &z1 + &z2;
    println!("z₁ + z₂ = {}", sum.eval());

    // z1 * z2 = (2+3I)(1-2I) = 2 - 4I + 3I - 6I² = 2 - I + 6 = 8 - I
    let product = (&z1 * &z2).expand().eval();
    println!("z₁ · z₂ = {product}");

    // ── 3. Euler's formula ─────────────────────────────────────────
    //
    // exp(I·x) = cos(x) + I·sin(x)
    //
    // This is one of the most beautiful results in mathematics,
    // connecting the exponential function to trigonometry via
    // the imaginary unit.

    println!("\n--- Euler's Formula ---");

    let eix = (&i * &x).exp();
    println!("exp(I·x) = {eix}");

    // Rewrite exp(I·x) in terms of trig functions
    let as_trig = eix.rewrite_as_trig();
    println!("Rewritten as trig: {as_trig}");

    // Euler's identity: exp(I·π) + 1 = 0
    let pi = ctx.pi();
    let euler_identity = &(&i * &pi).exp() + 1;
    let euler_simplified = euler_identity.eval().simplify();
    println!("\nexp(I·π) + 1 = {euler_simplified}");
    println!("  (Euler's identity: should equal 0)");

    // exp(I·π/2) = I
    let half_pi = &pi / 2;
    let exp_i_half_pi = (&i * &half_pi).exp().eval();
    println!("exp(I·π/2) = {exp_i_half_pi}");

    // ── 4. Complex quadratic roots ─────────────────────────────────
    //
    // x² + 1 = 0 has roots x = ±I

    println!("\n--- Complex Quadratic Roots ---");

    let eq1 = expr!(ctx, x ^ 2 + 1);
    let roots = eq1.solve_or_empty(&x);
    println!("x² + 1 = 0:");
    for r in &roots {
        println!("  x = {r}");
    }

    // x² + 2x + 5 = 0 → x = -1 ± 2I
    let eq2 = expr!(ctx, x ^ 2 + 2 * x + 5);
    let roots2 = eq2.solve_or_empty(&x);
    println!("\nx² + 2x + 5 = 0:");
    for r in &roots2 {
        println!("  x = {r}");
    }

    // x² - 2x + 2 = 0 → x = 1 ± I
    let eq3 = expr!(ctx, x ^ 2 - 2 * x + 2);
    let roots3 = eq3.solve_or_empty(&x);
    println!("\nx² - 2x + 2 = 0:");
    for r in &roots3 {
        println!("  x = {r}");
    }

    // x⁴ - 1 = 0 → x = 1, -1, I, -I
    let eq4 = expr!(ctx, x ^ 4 - 1);
    let roots4 = eq4.solve_or_empty(&x);
    println!("\nx⁴ - 1 = 0:");
    for r in &roots4 {
        println!("  x = {r}");
    }

    // ── 5. Numerical complex evaluation ────────────────────────────
    //
    // eval_complex64() returns a `Complex64` (num_complex) with `re`/`im` fields

    println!("\n--- Numerical Complex Evaluation ---");

    // I itself
    let Complex64 { re, im } = i.eval_complex64().unwrap();
    println!("I  → ({re}, {im})");

    // I²
    let Complex64 { re, im } = i.powi(2).eval().eval_complex64().unwrap();
    println!("I² → ({re}, {im})");

    // 2 + 3I
    if let Ok(Complex64 { re, im }) = z1.eval_complex64() {
        println!("2 + 3I → ({re}, {im})");
    }

    // sqrt(-1) = I
    let sqrt_neg1 = ctx.int(-1).sqrt();
    println!("\nsqrt(-1) = {sqrt_neg1}");
    if let Ok(Complex64 { re, im }) = sqrt_neg1.eval_complex64() {
        println!("  Numerical: ({re:.4}, {im:.4})");
    }

    // Evaluate complex roots numerically
    println!("\nNumerical values of roots of x² + 2x + 5 = 0:");
    for r in &roots2 {
        if let Ok(Complex64 { re, im }) = r.eval_complex64() {
            if im.abs() > 1e-10 {
                println!("  {re:.6} + {im:.6}i");
            } else {
                println!("  {re:.6}");
            }
        }
    }

    // ── 6. Trig-hyperbolic bridge ──────────────────────────────────
    //
    // cos(I·x) = cosh(x)
    // sin(I·x) = I·sinh(x)
    //
    // This connection arises from Euler's formula and links
    // circular and hyperbolic functions via the imaginary unit.

    println!("\n--- Trig–Hyperbolic Bridge ---");

    // Rewrite sin(x) and cos(x) as exponentials
    let sin_as_exp = x.sin().rewrite_as_exp();
    println!("sin(x) as exp: {sin_as_exp}");

    let cos_as_exp = x.cos().rewrite_as_exp();
    println!("cos(x) as exp: {cos_as_exp}");

    // Verify cos(I·x) at a specific value
    // cos(I·2) should equal cosh(2) ≈ 3.7622
    let ix2 = &i * 2;
    let cos_i2 = ix2.cos().eval();
    println!("\ncos(2I) = {cos_i2}");
    if let Ok(Complex64 { re, im }) = cos_i2.eval_complex64() {
        println!("  Numerical: ({re:.6}, {im:.6})");
        println!("  cosh(2)  = {:.6}", 2.0_f64.cosh());
    }

    // ── 7. Complex exponential ─────────────────────────────────────
    println!("\n--- Complex Exponential ---");

    // exp(1 + I·π) = e · exp(I·π) = e · (-1) = -e
    let z = &ctx.int(1) + &(&i * &pi);
    let exp_z = z.exp();
    println!("exp(1 + I·π) = {exp_z}");
    let simplified = exp_z.eval().simplify();
    println!("  Simplified: {simplified}");

    // exp(0) = 1
    let exp_0 = ctx.int(0).exp().eval();
    println!("exp(0) = {exp_0}");

    println!("\n✓ Done!");
}
