//! Matrix Algebra — eigenvalues, inverse, determinant, and more.
//!
//! Run with: cargo run --example matrix_algebra

use symplex::prelude::*;
use symplex::vars;

fn main() {
    println!("=== Matrix Algebra ===\n");

    vars!(x);

    // Numeric matrix
    let a = matrix![[2, 1], [1, 3]];
    println!("A = {a}");
    println!("det(A) = {}", a.det());
    println!("trace(A) = {}", a.trace());

    // Eigenvalues
    let eigenvals = a.eigenvals(&x);
    println!(
        "Eigenvalues: {:?}",
        eigenvals
            .iter()
            .map(|e| format!("{e}"))
            .collect::<Vec<_>>()
    );

    // Inverse
    if let Some(inv) = a.inv() {
        println!("A⁻¹ = {inv}");
        println!("A·A⁻¹ = {}", &a * &inv);
    }

    // Characteristic polynomial
    println!("Characteristic poly: {}", a.char_poly(&x));

    // Symbolic matrix
    let b = matrix![[x, 1], [0, x]];
    println!("\nB = {b}");
    println!("det(B) = {}", b.det());
    println!("B² = {}", &b * &b);

    // LaTeX
    println!("\nLaTeX:\n  {}", a.to_latex());

    println!("\n✓ Done!");
}
