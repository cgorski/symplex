//! Polynomial System Solving — find intersections of curves using Gröbner bases.
//!
//! Run with: cargo run --example solve_system

use symplex::prelude::*;


fn main() {
    println!("=== Polynomial System Solving ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    // Circle and line intersection
    println!("--- Circle ∩ Line ---");
    let solutions = symplex::polysys::solve_system_ex(
        &[expr!(x ^ 2 + y ^ 2 - 1), expr!(x + y - 1)],
        &[x.clone(), y.clone()],
    )
    .unwrap();

    for (i, sol) in solutions.iter().enumerate() {
        println!("  Solution {}: x = {}, y = {}", i + 1, sol[0], sol[1]);
    }

    // Two conics
    println!("\n--- Two Conics ---");
    let solutions = symplex::polysys::solve_system_ex(
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
    let solutions = symplex::polysys::solve_system_ex(
        &[expr!(x ^ 3 - 6 * x ^ 2 + 11 * x - 6)],
        std::slice::from_ref(&x),
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
