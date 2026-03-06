//! Symplex Quick Start — a tour of symbolic math in Rust.
//!
//! This example walks through the core capabilities of symplex:
//! expression building, differentiation, integration, simplification,
//! equation solving, factoring, matrix algebra, LaTeX output, and
//! code generation.
//!
//! Run with: cargo run --example quickstart

use symplex::prelude::*;
use symplex::vars;

fn main() {
    println!("=== Symplex Quick Start ===\n");

    vars!(x, y);

    // ── 1. Building expressions ────────────────────────────────────
    println!("--- Expression Building ---");
    let f = expr!(x ^ 2 + 2 * x + 1);
    println!("f(x) = {f}");

    // Exact rationals — no floating-point approximation
    let half = expr!(1 / 2);
    println!("1/2 = {half}");

    let g = &half * &x;
    println!("(1/2)*x = {g}");

    // ── 2. Differentiation ─────────────────────────────────────────
    println!("\n--- Differentiation ---");
    let df = f.diff(&x);
    println!("f'(x)  = {df}");

    let d2f = df.diff(&x);
    println!("f''(x) = {d2f}");

    // Partial derivatives
    let h = expr!(x ^ 2 * y + y ^ 3);
    println!("\nh(x,y) = {h}");
    println!("∂h/∂x  = {}", h.diff(&x));
    println!("∂h/∂y  = {}", h.diff(&y));

    // Higher-order derivative
    let p = expr!(x ^ 5);
    println!("\nd⁴/dx⁴ (x⁵) = {}", p.diff_n(&x, 4));

    // ── 3. Integration ─────────────────────────────────────────────
    println!("\n--- Integration ---");
    let anti = expr!(x ^ 2).integrate(&x);
    println!("∫ x² dx = {anti}");

    let poly_anti = f.integrate(&x);
    println!("∫ f(x) dx = {poly_anti}");

    // Definite integral
    let zero = symplex::int(0);
    let one = symplex::int(1);
    let area = expr!(x ^ 2).definite_integral(&x, &zero, &one);
    println!("∫₀¹ x² dx = {area}");

    // ── 4. Simplification ──────────────────────────────────────────
    println!("\n--- Simplification ---");
    let trig = expr!(sin(x) ^ 2 + cos(x) ^ 2);
    println!("{trig} → {}", trig.simplify());

    let exp_ln = x.ln().exp();
    println!("exp(ln(x)) → {}", exp_ln.simplify());

    let complicated = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
    println!("(x+1)² - x² - 2x → {}", complicated.full_simplify());

    // Trig simplification
    let trig2 = expr!(sin(x) ^ 2 + cos(x) ^ 2 + x);
    println!("{trig2} → {}", trig2.simplify_trig());

    // ── 5. Factoring ───────────────────────────────────────────────
    println!("\n--- Factoring ---");
    println!("x² - 1 = {}", expr!(x ^ 2 - 1).factor(&x));
    println!("x² - 5x + 6 = {}", expr!(x ^ 2 - 5 * x + 6).factor(&x));
    println!("x⁴ - 1 = {}", expr!(x ^ 4 - 1).factor(&x));

    // ── 6. Equation Solving ────────────────────────────────────────
    println!("\n--- Equation Solving ---");
    let roots = expr!(x ^ 2 - 5 * x + 6).solve_or_empty(&x);
    println!(
        "x² - 5x + 6 = 0 → {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );

    let cubic_roots = expr!(x ^ 3 - 6 * x ^ 2 + 11 * x - 6).solve_or_empty(&x);
    println!(
        "x³ - 6x² + 11x - 6 = 0 → {:?}",
        cubic_roots
            .iter()
            .map(|r| format!("{r}"))
            .collect::<Vec<_>>()
    );

    // ── 7. Numerical Evaluation ────────────────────────────────────
    println!("\n--- Numerical Evaluation ---");
    let val = expr!(sin(x) + cos(x)).eval_f64_with(&[(&x, 1)]).unwrap();
    println!("sin(1) + cos(1) = {val:.6}");

    let val2 = f.subs_i64(&x, 3);
    println!("f(3) = {val2}");

    // Arbitrary precision
    let pi = symplex::default_context().pi();
    if let Ok(s) = pi.eval_decimal(30) {
        println!("π to 30 digits: {s}");
    }

    // ── 8. Matrix Algebra ──────────────────────────────────────────
    println!("\n--- Matrix Algebra ---");
    let m = matrix![[2, 1], [1, 3]];
    println!("M = {m}");
    println!("det(M) = {}", m.det());
    println!("trace(M) = {}", m.trace());

    if let Some(inv) = m.inv() {
        println!("M⁻¹ = {inv}");
    }

    let eigenvals = m.eigenvals(&x);
    println!(
        "Eigenvalues: {:?}",
        eigenvals
            .iter()
            .map(|e| format!("{e}"))
            .collect::<Vec<_>>()
    );

    // Symbolic matrix
    let sym_m = matrix![[x, 1], [0, x]];
    println!("\nB = {sym_m}");
    println!("det(B) = {}", sym_m.det());

    // ── 9. LaTeX Output ────────────────────────────────────────────
    println!("\n--- LaTeX Output ---");
    println!("f(x):    {}", f.to_latex());
    println!("f'(x):   {}", df.to_latex());
    println!("sin²(x): {}", expr!(sin(x) ^ 2).to_latex());
    println!("Matrix:  {}", m.to_latex());

    // ── 10. Limits ─────────────────────────────────────────────────
    println!("\n--- Limits ---");
    let limit_expr = &x.sin() / &x;
    let lim = limit_expr.limit(&x, &symplex::int(0)).unwrap();
    println!("lim(x→0) sin(x)/x = {lim}");

    // ── 11. Series Expansion ───────────────────────────────────────
    println!("\n--- Series Expansion ---");
    let sin_series = x.sin().maclaurin(&x, 5).unwrap();
    println!("sin(x) ≈ {}", sin_series.expand().eval());

    // ── 12. Code Generation ────────────────────────────────────────
    println!("\n--- Code Generation ---");
    let code = df.to_rust_fn("f_prime", &["x"]).unwrap();
    println!("Generated Rust function:");
    println!("{code}");

    // ── 13. Compiled Function ──────────────────────────────────────
    println!("--- Compiled Evaluation ---");
    if let Some(compiled) = df.compile(&["x"]) {
        for val in [0.0, 1.0, 2.0, 3.0] {
            println!("  f'({val}) = {:.4}", compiled(&[val]));
        }
    }

    println!("\n✓ Done! See the tutorial at docs/tutorial/ for much more.");
}
