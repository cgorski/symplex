//! Gradient Descent — derive gradients symbolically, compile to fast Rust, optimize numerically.
//!
//! This example demonstrates the symplex "symbolic → compiled" pipeline:
//!
//!   1. Define a loss function as a symbolic expression
//!   2. Compute the gradient symbolically (exact, no finite differences)
//!   3. Compile the gradient to a native Rust closure via `compile()`
//!   4. Run gradient descent using the compiled closure (fast inner loop)
//!   5. Find the minimum
//!
//! This is the workflow ML engineers and optimization researchers want:
//! automatic differentiation with zero runtime overhead in the hot loop.
//!
//! Run with: `cargo run --example gradient_descent`

use symplex::prelude::*;

fn main() {
    println!("=== Gradient Descent with Symbolic Gradients ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    // ── 1. Define a loss function ──────────────────────────────────
    //
    // Rosenbrock function: f(x, y) = (1 - x)² + 100·(y - x²)²
    // Global minimum at (1, 1) with f(1, 1) = 0.
    // This is a classic benchmark for optimization algorithms.

    let f = expr!(ctx, (1 - x) ^ 2 + 100 * (y - x ^ 2) ^ 2);
    println!("Loss function:");
    println!("  f(x, y) = {f}\n");

    // ── 2. Compute the gradient symbolically ───────────────────────
    //
    // ∇f = [∂f/∂x, ∂f/∂y]
    // symplex computes the exact symbolic derivative — no finite
    // differences, no approximation errors.

    let df_dx = f.diff(&x).full_simplify();
    let df_dy = f.diff(&y).full_simplify();

    println!("Symbolic gradient:");
    println!("  ∂f/∂x = {df_dx}");
    println!("  ∂f/∂y = {df_dy}");

    // Verify: at the minimum (1, 1), both partials should be 0
    let grad_at_min_x = df_dx.eval_f64_with(&[(&x, 1), (&y, 1)]).unwrap();
    let grad_at_min_y = df_dy.eval_f64_with(&[(&x, 1), (&y, 1)]).unwrap();
    println!("\nGradient at known minimum (1, 1):");
    println!("  ∂f/∂x(1,1) = {grad_at_min_x:.6}");
    println!("  ∂f/∂y(1,1) = {grad_at_min_y:.6}");
    assert!(
        grad_at_min_x.abs() < 1e-10 && grad_at_min_y.abs() < 1e-10,
        "gradient should be zero at minimum"
    );

    // ── 3. Compile to fast closures ────────────────────────────────
    //
    // `compile()` turns the symbolic expression into a native Rust
    // closure that evaluates in nanoseconds — no arena lookup, no
    // tree traversal, just arithmetic.

    println!("\n--- Compiling gradient to native Rust closures ---");
    let grad_x_fn = df_dx.compile(&["x", "y"]).expect("compile ∂f/∂x");
    let grad_y_fn = df_dy.compile(&["x", "y"]).expect("compile ∂f/∂y");
    let loss_fn = f.compile(&["x", "y"]).expect("compile f");

    // Quick test: f(0, 0) = (1-0)² + 100·(0-0)² = 1
    println!("  f(0, 0)    = {:.4}", loss_fn(&[0.0, 0.0]));
    println!("  f(1, 1)    = {:.4}", loss_fn(&[1.0, 1.0]));
    println!("  f(-1, 2)   = {:.4}", loss_fn(&[-1.0, 2.0]));

    // ── 4. Run gradient descent ────────────────────────────────────
    //
    // Starting from (-1, 2), step toward the minimum at (1, 1).
    // We use a small learning rate because Rosenbrock is notoriously
    // ill-conditioned (narrow curved valley).

    println!("\n--- Gradient Descent ---");
    let learning_rate = 0.001;
    let max_iters = 50_000;
    let tolerance = 1e-10;

    let mut px = -1.0_f64;
    let mut py = 2.0_f64;
    let mut best_loss = loss_fn(&[px, py]);

    println!("  Start:     ({px:.4}, {py:.4}), loss = {best_loss:.6}");

    let mut converged = false;
    for i in 1..=max_iters {
        let gx = grad_x_fn(&[px, py]);
        let gy = grad_y_fn(&[px, py]);

        px -= learning_rate * gx;
        py -= learning_rate * gy;

        let loss = loss_fn(&[px, py]);

        // Print progress at exponential intervals
        if i <= 10 || i % 10_000 == 0 || (loss - best_loss).abs() < tolerance {
            if i <= 10 || i % 10_000 == 0 {
                println!("  Iter {i:>5}:  ({px:.4}, {py:.4}), loss = {loss:.6}");
            }
        }

        if (best_loss - loss).abs() < tolerance && loss < 1e-6 {
            println!("  Iter {i:>5}:  ({px:.4}, {py:.4}), loss = {loss:.10}");
            println!("  Converged after {i} iterations!");
            converged = true;
            best_loss = loss;
            break;
        }
        best_loss = loss;
    }

    if !converged {
        println!("  Final:     ({px:.6}, {py:.6}), loss = {best_loss:.10}");
        println!("  (Did not fully converge in {max_iters} iterations — Rosenbrock is hard!)");
    }

    // ── 5. Verify the result ───────────────────────────────────────

    println!("\n--- Verification ---");
    println!("  Known minimum:  (1.0000, 1.0000)");
    println!("  Found:          ({px:.4}, {py:.4})");
    println!("  Distance:       {:.6}", ((px - 1.0).powi(2) + (py - 1.0).powi(2)).sqrt());
    println!("  Final loss:     {best_loss:.10}");

    // ── 6. Show the generated code ─────────────────────────────────
    //
    // For users who want to see what the compiled gradient looks like
    // as a Rust function (e.g., for embedding in a project):

    println!("\n--- Generated Rust Code for ∂f/∂x ---");
    match df_dx.to_rust_fn("grad_x", &["x", "y"]) {
        Ok(code) => println!("{code}"),
        Err(_) => println!("  (expression too complex for single-function codegen)"),
    }

    println!("\n✓ Done!");
}
