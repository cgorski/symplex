//! LaTeX Output — render symbolic math as publication-quality LaTeX.
//!
//! Run with: cargo run --example latex_output

use symplex::prelude::*;
use symplex::vars;

fn main() {
    println!("=== LaTeX Output ===\n");

    vars!(x, y);

    let exprs: Vec<(&str, Ex)> = vec![
        ("Quadratic", expr!(x ^ 2 + 2 * x + 1)),
        ("Fraction", symplex::rational(3, 7)),
        ("Trig", expr!(sin(x) ^ 2 + cos(x) ^ 2)),
        ("Derivative", expr!(x ^ 3 + x).diff(&x)),
        ("Square root", expr!(x).sqrt()),
    ];

    for (name, e) in &exprs {
        println!("{name}:");
        println!("  Display: {e}");
        println!("  LaTeX:   {}", e.to_latex());
        println!();
    }

    // Integral rendered as LaTeX
    let anti = expr!(x ^ 2).integrate(&x);
    println!("Integral:");
    println!("  Display: {anti}");
    println!("  LaTeX:   {}", anti.to_latex());
    println!();

    // Expression involving y
    let multi = &x.powi(2) + &y.powi(2);
    println!("Multivariate:");
    println!("  Display: {multi}");
    println!("  LaTeX:   {}", multi.to_latex());
    println!();

    // Matrix LaTeX
    let m = matrix![[1, 2], [3, 4]];
    println!("Matrix:");
    println!("  Display: {m}");
    println!("  LaTeX:\n  {}", m.to_latex());
    println!();

    // Inline and display modes
    let f = expr!(x ^ 2 + 1);
    println!("Inline:  {}", f.to_latex_inline());
    println!("Display: {}", f.to_latex_display());

    println!("\n✓ Done!");
}
