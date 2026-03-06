//! Symplex Quick Start — symbolic math in Rust.
//!
//! Run with: cargo run --example quickstart

use symplex::prelude::*;
use symplex::vars;

fn main() {
    vars!(x, y);

    let f = expr!(x ^ 2 + 2 * x + 1);
    println!("f(x) = {f}");
    println!("f'(x) = {}", f.diff(&x));

    let anti = expr!(x ^ 2).integrate(&x);
    println!("∫ x² dx = {anti}");

    println!("sin²+cos² = {}", expr!(sin(x) ^ 2 + cos(x) ^ 2).simplify());

    let roots = expr!(x ^ 2 - 5 * x + 6).solve_or_empty(&x);
    println!(
        "roots: {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );

    println!(
        "sin(1)+cos(1) = {:.6}",
        expr!(sin(x) + cos(x)).eval_f64_with(&[(&x, 1)]).unwrap()
    );

    let m = matrix![[1, 2], [3, 4]];
    println!("det = {}", m.det());

    println!("LaTeX: {}", f.diff(&x).to_latex());

    // Partial derivatives
    let f2 = expr!(x ^ 2 * y + y ^ 3);
    println!("∂f/∂x = {}", f2.diff(&x));
    println!("∂f/∂y = {}", f2.diff(&y));

    println!("\n✓ Done!");
}
