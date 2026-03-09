//! Equation Solving — algebraic, transcendental, and numerical.
//!
//! Demonstrates:
//! - Quadratic, cubic, and quartic polynomial solving
//! - Complex roots
//! - Transcendental equations (sin(x) = 1/2)
//! - Polynomial system solving via Gröbner bases
//! - Factoring
//! - Inequality solving
//! - Numerical root finding (Newton's method)
//! - Solution verification
//!
//! Run with: cargo run --example equation_solving

use symplex::prelude::*;


fn main() {
    println!("=== Equation Solving ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    // ── Single-variable polynomial equations ───────────────────────

    println!("--- Quadratic ---");
    let quadratic = expr!(x ^ 2 - 5 * x + 6);
    let roots = quadratic.solve_or_empty(&x);
    println!(
        "x² - 5x + 6 = 0 → x ∈ {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );

    // Verify each root
    for r in &roots {
        let verified = quadratic.check_solution(&x, r) == Some(true);
        println!("  x = {r}: verified = {verified}");
    }

    println!("\n--- Cubic ---");
    let cubic = expr!(x ^ 3 - 6 * x ^ 2 + 11 * x - 6);
    let roots = cubic.solve_or_empty(&x);
    println!(
        "x³ - 6x² + 11x - 6 = 0 → x ∈ {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
    // Should be 1, 2, 3
    for r in &roots {
        let verified = cubic.check_solution(&x, r) == Some(true);
        println!("  x = {r}: verified = {verified}");
    }

    println!("\n--- Quartic ---");
    let quartic = expr!(x ^ 4 - 5 * x ^ 2 + 4);
    let roots = quartic.solve_or_empty(&x);
    println!(
        "x⁴ - 5x² + 4 = 0 → x ∈ {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
    // Should include ±1, ±2

    println!("\n--- Quartic with complex roots ---");
    let roots = expr!(x ^ 4 - 1).solve_or_empty(&x);
    println!(
        "x⁴ - 1 = 0 → x ∈ {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
    // Real roots: 1, -1; complex roots: I, -I
    println!("  (Real roots: ±1; complex roots: ±I)");

    // Evaluate complex roots numerically
    for r in &roots {
        if let Ok((re, im)) = r.eval_complex64()
            && im.abs() > 1e-10
        {
            println!("  Complex root: {re:.4} + {im:.4}i");
        }
    }

    println!("\n--- Quadratic with only complex roots ---");
    let complex_quad = expr!(x ^ 2 + 1);
    let roots = complex_quad.solve_or_empty(&x);
    println!(
        "x² + 1 = 0 → x ∈ {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
    println!("  (Roots are ±I — the imaginary unit)");

    // ── Transcendental equation ─────────────────────────────────────

    println!("\n--- Transcendental: sin(x) = 1/2 ---");
    // Rewrite as sin(x) - 1/2 = 0
    let eq = &x.sin() - &ctx.rational(1, 2);
    let roots = eq.solve_or_empty(&x);
    if roots.is_empty() {
        println!("  No symbolic roots found (expected for transcendental)");
        println!("  Falling back to numerical solve...");
        // Try numerical: we know a root near π/6 ≈ 0.5236
        match eq.solve_numeric(&x, 0.5, 50, 1e-12) {
            Ok(root) => {
                println!("  Numerical root: x ≈ {root:.10}");
                println!("  (π/6 ≈ {:.10})", std::f64::consts::FRAC_PI_6);
            }
            Err(e) => println!("  Numerical solve failed: {e}"),
        }
    } else {
        for r in &roots {
            println!("  x = {r}");
        }
    }

    // ── Set-valued solutions ───────────────────────────────────────

    println!("\n--- Set-Valued Solutions ---");
    let result = expr!(x ^ 2 - 5 * x + 6).solve_as_set(&x);
    println!("x² - 5x + 6 = 0 as set: {result}");

    // ── Polynomial system (Gröbner bases) ──────────────────────────

    println!("\n--- System: Circle ∩ Line ---");
    let eq1 = expr!(x ^ 2 + y ^ 2 - 1);
    let eq2 = expr!(x + y - 1);
    let solutions = symplex::polysys::solve_system_ex(
        &[eq1.clone(), eq2.clone()],
        &[x.clone(), y.clone()],
    )
    .unwrap();
    for (i, sol) in solutions.iter().enumerate() {
        println!("  Solution {}: x = {}, y = {}", i + 1, sol[0], sol[1]);
    }

    // Verify each solution against both equations
    println!("  Verification:");
    for (i, sol) in solutions.iter().enumerate() {
        let r1 = eq1.subs(&x, &sol[0]).subs(&y, &sol[1]).eval();
        let r2 = eq2.subs(&x, &sol[0]).subs(&y, &sol[1]).eval();
        println!("    Sol {}: eq1 residual = {r1}, eq2 residual = {r2}", i + 1);
    }

    println!("\n--- System: Two Conics ---");
    let solutions = symplex::polysys::solve_system_ex(
        &[expr!(x ^ 2 + y ^ 2 - 5), expr!(x * y - 2)],
        &[x.clone(), y.clone()],
    )
    .unwrap();
    println!("  x² + y² = 5, xy = 2:");
    println!("  Found {} solutions:", solutions.len());
    for (i, sol) in solutions.iter().enumerate() {
        println!("    Solution {}: x = {}, y = {}", i + 1, sol[0], sol[1]);
    }

    // ── System with irrational roots ───────────────────────────────

    println!("\n--- System with Irrational Roots ---");
    // x² + y² = 3, x + y = 1
    // Solutions involve √-expressions
    match symplex::polysys::solve_system_ex(
        &[expr!(x ^ 2 + y ^ 2 - 3), expr!(x + y - 1)],
        &[x.clone(), y.clone()],
    ) {
        Ok(solutions) => {
            println!("  x² + y² = 3, x + y = 1:");
            for (i, sol) in solutions.iter().enumerate() {
                println!("    Solution {}: x = {}, y = {}", i + 1, sol[0], sol[1]);
            }
        }
        Err(e) => {
            println!("  Gröbner solver returned error: {e}");
            println!("  (This can happen when roots are irrational)");
        }
    }

    // ── Factoring ──────────────────────────────────────────────────

    println!("\n--- Factoring ---");
    println!("x² - 1       = {}", expr!(x ^ 2 - 1).factor(&x));
    println!("x² + 2x + 1  = {}", expr!(x ^ 2 + 2 * x + 1).factor(&x));
    println!("x³ - 1        = {}", expr!(x ^ 3 - 1).factor(&x));
    println!("x⁴ - 1        = {}", expr!(x ^ 4 - 1).factor(&x));
    println!(
        "x² - 5x + 6   = {}",
        expr!(x ^ 2 - 5 * x + 6).factor(&x)
    );

    // x² + 1 has no real factors
    let no_factor = expr!(x ^ 2 + 1).factor(&x);
    println!("x² + 1        = {no_factor} (no real factors)");

    // ── Inequality solving ─────────────────────────────────────────

    println!("\n--- Inequality Solving ---");

    // x² - 4 > 0 → x < -2 or x > 2
    let result = expr!(x ^ 2 - 4).solve_gt(&x);
    println!("x² - 4 > 0:  {result}");

    // x² - 4 >= 0
    let result = expr!(x ^ 2 - 4).solve_ge(&x);
    println!("x² - 4 ≥ 0:  {result}");

    // x² - 4 < 0 → -2 < x < 2
    let result = expr!(x ^ 2 - 4).solve_lt(&x);
    println!("x² - 4 < 0:  {result}");

    // x² - 4 <= 0 → -2 <= x <= 2
    let result = expr!(x ^ 2 - 4).solve_le(&x);
    println!("x² - 4 ≤ 0:  {result}");

    // x > 0
    let result = x.solve_gt(&x);
    println!("x > 0:        {result}");

    // ── Numerical root finding ─────────────────────────────────────

    println!("\n--- Numerical Root Finding ---");

    // x = cos(x) — the Dottie number
    let f_transcendental = &x - &x.cos();
    println!("Solving x - cos(x) = 0 (Newton's method):");
    match f_transcendental.solve_numeric(&x, 1.0, 50, 1e-12) {
        Ok(root) => {
            println!("  x ≈ {root:.12}");
            // Verify: root - cos(root) should be ~0
            let residual = root - root.cos();
            println!("  Residual: {residual:.2e}");
        }
        Err(e) => println!("  Failed: {e}"),
    }

    // exp(x) = 3 → x = ln(3)
    let exp_eq = &x.exp() - 3;
    println!("\nSolving exp(x) = 3:");
    match exp_eq.solve_numeric(&x, 1.0, 50, 1e-12) {
        Ok(root) => {
            println!("  x ≈ {root:.12}");
            println!("  ln(3) = {:.12}", 3.0_f64.ln());
        }
        Err(e) => println!("  Failed: {e}"),
    }

    // Find multiple roots of x³ - 6x² + 11x - 6 = 0 numerically
    let poly = expr!(x ^ 3 - 6 * x ^ 2 + 11 * x - 6);
    println!("\nFinding roots of x³ - 6x² + 11x - 6 = 0 numerically:");
    for guess in [0.5, 1.5, 3.5] {
        match poly.solve_numeric(&x, guess, 50, 1e-12) {
            Ok(root) => println!("  guess {guess} → root {root:.8}"),
            Err(e) => println!("  guess {guess} → error: {e}"),
        }
    }

    // ── Symbolic vs Numerical comparison ───────────────────────────

    println!("\n--- Symbolic vs Numerical ---");
    let eq = expr!(x ^ 2 - 2);
    let sym_roots = eq.solve_or_empty(&x);
    println!("x² - 2 = 0:");
    println!(
        "  Symbolic: {:?}",
        sym_roots
            .iter()
            .map(|r| format!("{r}"))
            .collect::<Vec<_>>()
    );
    println!("  (These are exact — involving √2)");

    // Numerical comparison
    for r in &sym_roots {
        if let Ok(v) = r.eval_f64() {
            println!("  Numerical value: {v:.12}");
        }
    }

    println!("\n✓ Done!");
}
