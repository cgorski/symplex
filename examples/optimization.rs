//! Symbolic Optimization — gradient, Hessian, critical points.
//!
//! Demonstrates how to use symplex for multivariable calculus tasks
//! common in optimization: computing gradients, Hessians, finding
//! critical points, and classifying them via the second-derivative test.
//!
//! Run with: cargo run --example optimization

use symplex::prelude::*;


fn main() {
    println!("=== Symbolic Optimization ===\n");

    let __ctx = symplex::default_context();
    symplex::syms!(__ctx; x, y);

    // Function: f(x,y) = x² + y² - 2x - 4y + 5
    let f = expr!(x ^ 2 + y ^ 2 - 2 * x - 4 * y + 5);
    println!("f(x,y) = {f}");

    // Gradient: ∇f = [∂f/∂x, ∂f/∂y]
    let grad_x = f.diff(&x);
    let grad_y = f.diff(&y);
    println!("∇f = [{grad_x}, {grad_y}]");

    // Critical points: ∇f = 0
    let x_crit = grad_x.solve_or_empty(&x);
    let y_crit = grad_y.solve_or_empty(&y);
    println!(
        "Critical point: x = {}, y = {}",
        x_crit
            .first()
            .map(|v| format!("{v}"))
            .unwrap_or("?".into()),
        y_crit
            .first()
            .map(|v| format!("{v}"))
            .unwrap_or("?".into()),
    );

    // Hessian: H = [[∂²f/∂x², ∂²f/∂x∂y], [∂²f/∂y∂x, ∂²f/∂y²]]
    let fxx = f.diff(&x).diff(&x);
    let fxy = f.diff(&x).diff(&y);
    let fyx = f.diff(&y).diff(&x);
    let fyy = f.diff(&y).diff(&y);
    let hessian = matrix![
        [fxx, fxy],
        [fyx, fyy]
    ];
    println!("Hessian: {hessian}");
    println!("det(H) = {}", hessian.det().unwrap());

    // Classification: det(H) > 0 and ∂²f/∂x² > 0 → local minimum
    let det_h = hessian
        .det()
        .unwrap()
        .eval_f64_with(&[(&x, 1), (&y, 2)])
        .unwrap();
    let fxx = hessian
        .get(0, 0)
        .eval_f64_with(&[(&x, 1), (&y, 2)])
        .unwrap();
    println!("At critical point: det(H) = {det_h}, f_xx = {fxx}");
    if det_h > 0.0 && fxx > 0.0 {
        println!("→ Local MINIMUM");
    } else if det_h > 0.0 && fxx < 0.0 {
        println!("→ Local MAXIMUM");
    } else if det_h < 0.0 {
        println!("→ Saddle point");
    } else {
        println!("→ Inconclusive");
    }

    println!("\n✓ Done!");
}
