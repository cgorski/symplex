//! Equation Solving — algebraic and polynomial systems.
//!
//! Demonstrates solving quadratic, cubic, and quartic equations,
//! polynomial system solving via Gröbner bases, and factoring.
//!
//! Run with: cargo run --example equation_solving

use symplex::prelude::*;
use symplex::vars;

fn main() {
    println!("=== Equation Solving ===\n");

    vars!(x, y);

    // ── Single-variable equations ──────────────────────────────────────

    println!("--- Quadratic ---");
    let roots = expr!(x ^ 2 - 5 * x + 6).solve_or_empty(&x);
    println!(
        "x² - 5x + 6 = 0 → x ∈ {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );

    println!("\n--- Cubic ---");
    let roots = expr!(x ^ 3 - 6 * x ^ 2 + 11 * x - 6).solve_or_empty(&x);
    println!(
        "x³ - 6x² + 11x - 6 = 0 → x ∈ {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );

    println!("\n--- Quartic with complex roots ---");
    let roots = expr!(x ^ 4 - 1).solve_or_empty(&x);
    println!(
        "x⁴ - 1 = 0 → x ∈ {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );

    // ── Polynomial system (Gröbner bases) ──────────────────────────────

    println!("\n--- System: Circle ∩ Line ---");
    let solutions = symplex::solve_system(
        &[expr!(x ^ 2 + y ^ 2 - 1), expr!(x + y - 1)],
        &[x.clone(), y.clone()],
    )
    .unwrap();
    for (i, sol) in solutions.iter().enumerate() {
        println!("  Solution {}: x = {}, y = {}", i + 1, sol[0], sol[1]);
    }

    // ── Factoring ──────────────────────────────────────────────────────

    println!("\n--- Factoring ---");
    println!("x⁴ - 1 = {}", expr!(x ^ 4 - 1).factor(&x));
    println!("x² + 2x + 1 = {}", expr!(x ^ 2 + 2 * x + 1).factor(&x));

    println!("\n✓ Done!");
}
