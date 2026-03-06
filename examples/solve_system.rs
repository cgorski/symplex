//! Polynomial System Solving — find intersections of curves using Gröbner bases.
//!
//! Run with: cargo run --example solve_system

use symplex::prelude::*;
use symplex::vars;

fn main() {
    println!("=== Polynomial System Solving ===\n");

    vars!(x, y);

    // Circle and line intersection
    println!("--- Circle ∩ Line ---");
    let solutions = symplex::solve_system(
        &[expr!(x ^ 2 + y ^ 2 - 1), expr!(x + y - 1)],
        &[x.clone(), y.clone()],
    )
    .unwrap();

    for (i, sol) in solutions.iter().enumerate() {
        println!("  Solution {}: x = {}, y = {}", i + 1, sol[0], sol[1]);
    }

    // Two conics
    println!("\n--- Two Conics ---");
    let solutions = symplex::solve_system(
        &[expr!(x ^ 2 + y ^ 2 - 5), expr!(x * y - 2)],
        &[x.clone(), y.clone()],
    )
    .unwrap();

    println!("  Found {} solutions:", solutions.len());
    for (i, sol) in solutions.iter().enumerate() {
        println!("  Solution {}: x = {}, y = {}", i + 1, sol[0], sol[1]);
    }

    // Cubic system (univariate)
    println!("\n--- Cubic ---");
    let solutions = symplex::solve_system(
        &[expr!(x ^ 3 - 6 * x ^ 2 + 11 * x - 6)],
        &[x.clone()],
    )
    .unwrap();
    println!(
        "  x³-6x²+11x-6 = 0 → x ∈ {:?}",
        solutions
            .iter()
            .map(|s| format!("{}", s[0]))
            .collect::<Vec<_>>()
    );

    println!("\n✓ Done!");
}
