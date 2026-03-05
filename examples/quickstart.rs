//! Symplex Quick Start — symbolic math in Rust.
//!
//! Run with: cargo run --example quickstart

use symplex::prelude::*;
use symplex::vars;

fn main() {
    // Declare symbolic variables
    vars!(x, y);

    // Build expressions with natural math syntax
    let f = expr!(x ^ 2 + 2 * x + 1);
    println!("f(x) = {f}");

    // Differentiate
    let df = f.diff(&x);
    println!("f'(x) = {df}");

    // Integrate
    let anti = expr!(x ^ 2).integrate(&x);
    println!("∫ x² dx = {anti}");

    // Simplify trig identities
    let trig = expr!(sin(x) ^ 2 + cos(x) ^ 2);
    println!("sin²(x) + cos²(x) = {}", trig.simplify());

    // Solve equations
    let roots = expr!(x ^ 2 - 5 * x + 6).solve_or_empty(&x);
    println!(
        "x² - 5x + 6 = 0  →  x = {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );

    // Evaluate numerically
    let val = expr!(sin(x) + cos(x))
        .subs_i64(&x, 1)
        .eval()
        .evalf_f64()
        .unwrap();
    println!("sin(1) + cos(1) = {val:.6}");

    // Taylor series
    let series = x.sin().maclaurin(&x, 4).unwrap().expand();
    println!("sin(x) ≈ {series}");

    // Matrices
    let m = matrix![[1, 2], [3, 4]];
    println!("det = {}", m.det());

    // Use y so it's not unused
    let g = &x + &y;
    println!("x + y = {g}");

    println!("\n✓ All operations completed successfully!");
}
