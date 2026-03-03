//! Calculus workflow example for symplex.
//!
//! Demonstrates: expression building, differentiation, integration,
//! series expansion, equation solving, simplification, and numerical evaluation.
//!
//! Run with: `cargo run --example calculus`

use symplex::prelude::*;

fn main() {
    println!("=== Symplex Calculus Example ===\n");

    // ── 1. Create symbols ──────────────────────────────────────────────
    let x = symplex::var("x");

    // ── 2. Build a function and differentiate ──────────────────────────
    let f = expr!(x ^ 3 - 3 * x ^ 2 + 2 * x);
    println!("f(x)   = {f}");

    let df = f.diff(&x);
    println!("f'(x)  = {df}");

    let d2f = df.diff(&x);
    println!("f''(x) = {d2f}");

    // ── 3. Find critical points (where f'(x) = 0) ─────────────────────
    println!("\nCritical points (f'(x) = 0):");
    let critical = df.solve(&x).unwrap_or_default();
    for pt in &critical {
        println!("  x = {pt}");
    }

    // ── 4. Classify critical points using the second derivative ────────
    println!("\nSecond-derivative test:");
    for pt in &critical {
        let d2_val = d2f.subs(&x, pt);
        println!("  f''({pt}) = {d2_val}");
    }

    // ── 5. Integrate ───────────────────────────────────────────────────
    let anti = f.integrate(&x);
    println!("\n∫ f(x) dx = {anti}");

    // Verify: d/dx(∫ f dx) should give back f
    let roundtrip = anti.diff(&x);
    println!("d/dx(∫ f dx) = {roundtrip}");

    // ── 6. Definite integral ───────────────────────────────────────────
    let zero = symplex::int(0);
    let one = symplex::int(1);
    let area = f.definite_integral(&x, &zero, &one);
    println!("\n∫₀¹ f(x) dx = {area}");

    // ── 7. Taylor series of sin(x) around 0 ───────────────────────────
    let sin_series = x.sin().maclaurin(&x, 5).unwrap();
    println!("\nsin(x) ≈ {}", sin_series.expand().eval());

    let cos_series = x.cos().maclaurin(&x, 5).unwrap();
    println!("cos(x) ≈ {}", cos_series.expand().eval());

    // ── 8. Simplification: trig identity ───────────────────────────────
    let trig = expr!(sin(x) ^ 2 + cos(x) ^ 2);
    println!("\n{trig} → {}", trig.simplify());

    // ── 9. More simplification: exp/ln inverse ─────────────────────────
    let exp_ln = x.ln().exp();
    println!("exp(ln(x)) → {}", exp_ln.simplify());

    let ln_exp = x.exp().ln();
    println!("ln(exp(x)) → {}", ln_exp.simplify());

    // ── 10. Limits ─────────────────────────────────────────────────────
    let limit_expr = &x.sin() / &x;
    let lim = limit_expr.limit(&x, &symplex::int(0)).unwrap();
    println!("\nlim(x→0) sin(x)/x = {lim}");

    // ── 11. Numerical evaluation ───────────────────────────────────────
    let val = expr!(x ^ 2 + 1).subs_i64(&x, 3);
    println!("\nf(3) where f = x² + 1: {val}");

    let val2 = f.subs_i64(&x, 5);
    println!("f(5) where f = x³ - 3x² + 2x: {val2}");

    // ── 12. Factor a polynomial ────────────────────────────────────────
    let poly = &x.powi(2) - 1;
    let factored = poly.factor(&x);
    println!("\nx² - 1 = {factored}");

    let quadratic = expr!(x ^ 2 - 5 * x + 6);
    let factored2 = quadratic.factor(&x);
    println!("x² - 5x + 6 = {factored2}");

    // ── 13. Solve equations ────────────────────────────────────────────
    println!("\nSolving x² - 5x + 6 = 0:");
    let roots = quadratic.solve(&x).unwrap();
    for r in &roots {
        println!("  x = {r}");
    }

    // ── 14. Full simplify: expand then simplify ────────────────────────
    let complicated = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
    println!("\n(x+1)² - x² - 2x = {}", complicated.full_simplify());

    // ── 15. Numerical root finding ─────────────────────────────────────
    // Solve x = cos(x) numerically
    let transcendental = &x - &x.cos();
    match transcendental.nsolve(&x, 1.0, 50, 1e-12) {
        Ok(root) => println!("\nNumerical root of x - cos(x) = 0: x ≈ {root:.10}"),
        Err(e) => println!("\nNumerical solve failed: {e}"),
    }

    // ── 16. Arbitrary-precision evaluation ─────────────────────────────
    let pi = symplex::default_context().pi();
    match pi.evalf(30) {
        Ok(s) => println!("\nπ to 30 digits: {s}"),
        Err(e) => println!("\nevalf failed: {e}"),
    }

    println!("\n=== Done ===");
}
