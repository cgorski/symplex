//! ODE Solving — separable, linear, and second-order constant-coefficient.
//!
//! Demonstrates:
//! - Simple separable: y' = f(x)
//! - Exponential decay: y' + a·y = 0
//! - Full separable: y' = f(x)·g(y)
//! - Second-order homogeneous: y'' + b·y' + c·y = 0
//! - ODE classification
//! - Solution verification with check_ode_solution()
//!
//! Run with: cargo run --example ode_solving

use symplex::prelude::*;
use symplex::vars;

fn main() {
    println!("=== ODE Solving ===\n");

    vars!(x, y);

    // ── 1. Simple separable: y' = x ───────────────────────────────
    //
    // y' - x = 0  →  y = x²/2 + C1

    println!("--- Simple Separable: y' = x ---");
    let ode1 = expr!(diff(y, x) - x);
    println!("ODE: {ode1} = 0");
    println!("Type: {:?}", ode1.classify_ode(&y, &x));

    if let Some((sol, constants)) = ode1.solve_ode(&y, &x) {
        println!("Solution: y = {sol}");
        println!(
            "Constants: {:?}",
            constants.iter().map(|c| format!("{c}")).collect::<Vec<_>>()
        );

        // Verify: y = x²/2 satisfies y' - x = 0
        let particular = &x.powi(2) / 2;
        let verified = ode1.check_ode_solution(&particular, &y, &x);
        println!("Verify y = x²/2: {verified}");
    }

    // ── 2. Exponential decay: y' + 2y = 0 ─────────────────────────
    //
    // Solution: y = C1·exp(-2x)

    println!("\n--- First-Order Linear CC: y' + 2y = 0 ---");
    let ode2 = expr!(diff(y, x) + 2 * y);
    println!("ODE: {ode2} = 0");
    println!("Type: {:?}", ode2.classify_ode(&y, &x));

    if let Some((sol, _)) = ode2.solve_ode(&y, &x) {
        println!("Solution: y = {sol}");

        // Verify: y = exp(-2x) satisfies y' + 2y = 0
        let particular = (-&x * 2).exp();
        let verified = ode2.check_ode_solution(&particular, &y, &x);
        println!("Verify y = exp(-2x): {verified}");
    }

    // ── 3. y' = 0 (constant solution) ─────────────────────────────

    println!("\n--- Trivial: y' = 0 ---");
    let ode3 = y.formal_diff(&x);
    println!("ODE: {} = 0", ode3);

    if let Some((sol, _)) = ode3.solve_ode(&y, &x) {
        println!("Solution: y = {sol}");
    }

    // ── 4. Second-order homogeneous: y'' + y = 0 ──────────────────
    //
    // Characteristic equation: r² + 1 = 0 → r = ±i
    // General solution: y = C1·cos(x) + C2·sin(x)

    println!("\n--- Second-Order CC: y'' + y = 0 ---");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode4 = &d2y + &y;
    println!("ODE: {ode4} = 0");
    println!("Type: {:?}", ode4.classify_ode(&y, &x));

    if let Some((sol, constants)) = ode4.solve_ode(&y, &x) {
        println!("Solution: y = {sol}");
        println!(
            "Constants: {:?}",
            constants.iter().map(|c| format!("{c}")).collect::<Vec<_>>()
        );

        // Verify particular solutions
        let cos_sol = x.cos();
        let sin_sol = x.sin();
        println!("Verify y = cos(x): {}", ode4.check_ode_solution(&cos_sol, &y, &x));
        println!("Verify y = sin(x): {}", ode4.check_ode_solution(&sin_sol, &y, &x));

        // Wrong solution
        let wrong = x.exp();
        println!("Verify y = exp(x): {} (should be false)", ode4.check_ode_solution(&wrong, &y, &x));
    }

    // ── 5. Second-order with damping: y'' + 3y' + 2y = 0 ─────────
    //
    // Characteristic equation: r² + 3r + 2 = 0 → r = -1, -2
    // General solution: y = C1·exp(-x) + C2·exp(-2x)

    println!("\n--- Second-Order CC (overdamped): y'' + 3y' + 2y = 0 ---");
    let dy5 = y.formal_diff(&x);
    let d2y5 = dy5.formal_diff(&x);
    let ode5 = &(&d2y5 + &(&dy5 * 3)) + &(&y * 2);
    println!("ODE: {ode5} = 0");
    println!("Type: {:?}", ode5.classify_ode(&y, &x));

    if let Some((sol, _)) = ode5.solve_ode(&y, &x) {
        println!("Solution: y = {sol}");

        // Verify: y = exp(-x) should satisfy the ODE
        let exp_neg_x = (-&x).exp();
        let verified = ode5.check_ode_solution(&exp_neg_x, &y, &x);
        println!("Verify y = exp(-x): {verified}");

        // Verify: y = exp(-2x) should also satisfy the ODE
        let exp_neg_2x = (-&x * 2).exp();
        let verified2 = ode5.check_ode_solution(&exp_neg_2x, &y, &x);
        println!("Verify y = exp(-2x): {verified2}");
    }

    // ── 6. Full separable: y' = x·y ──────────────────────────────
    //
    // Separation of variables: dy/y = x dx → ln|y| = x²/2 + C
    // Solution: y = C1·exp(x²/2)

    println!("\n--- Full Separable: y' = x·y ---");
    let ode6 = &y.formal_diff(&x) - &(&x * &y);
    println!("ODE: y' - x·y = 0");
    println!("Type: {:?}", ode6.classify_ode(&y, &x));

    if let Some((sol, _)) = ode6.solve_ode(&y, &x) {
        println!("Solution: y = {sol}");
    } else {
        println!("Solver did not find a closed-form solution");
    }

    // ── 7. Using expr! macro for ODE construction ──────────────────

    println!("\n--- ODE via expr! Macro ---");
    let ode_macro = expr!(diff(y, x) + y);
    println!("expr!(diff(y, x) + y) = {ode_macro}");
    println!("Type: {:?}", ode_macro.classify_ode(&y, &x));

    if let Some((sol, _)) = ode_macro.solve_ode(&y, &x) {
        println!("Solution: y = {sol}");
    }

    // ── 8. Classification summary ──────────────────────────────────

    println!("\n--- Classification Summary ---");

    let odes: Vec<(&str, Ex)> = vec![
        ("y' = x", expr!(diff(y, x) - x)),
        ("y' + 2y = 0", expr!(diff(y, x) + 2 * y)),
        ("y' = x·y", {
            let dy = y.formal_diff(&x);
            &dy - &(&x * &y)
        }),
    ];

    for (desc, ode) in &odes {
        let classification = ode.classify_ode(&y, &x);
        let solvable = ode.solve_ode(&y, &x).is_some();
        println!("  {desc:30} → {:?} (solvable: {solvable})", classification);
    }

    println!("\n✓ Done!");
}
