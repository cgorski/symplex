//! Calculus workflow example for symplex.
//!
//! Demonstrates: expression building, differentiation, integration,
//! series expansion, equation solving, simplification, numerical evaluation,
//! ODE solving, and implicit differentiation.
//!
//! Run with: `cargo run --example calculus`

use symplex::prelude::*;
use symplex::vars;

fn main() {
    println!("=== Symplex Calculus Example ===\n");

    // ── 1. Create symbols ──────────────────────────────────────────────
    vars!(x);

    // ── 2. Build a function and differentiate ──────────────────────────
    let f = expr!(x ^ 3 - 3 * x ^ 2 + 2 * x);
    println!("f(x)   = {f}");

    let df = f.diff(&x);
    println!("f'(x)  = {df}");

    let d2f = df.diff(&x);
    println!("f''(x) = {d2f}");

    // Higher-order derivative
    let g = expr!(x ^ 6);
    let g4 = g.diff_n(&x, 4);
    println!("\nd⁴/dx⁴ (x⁶) = {g4}");

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

    // More integration examples
    println!("\nAdditional integrals:");
    println!("  ∫ sin(x) dx = {}", x.sin().integrate(&x));
    println!("  ∫ cos(x) dx = {}", x.cos().integrate(&x));
    println!("  ∫ exp(x) dx = {}", x.exp().integrate(&x));
    println!("  ∫ 1/x dx    = {}", (1 / &x).integrate(&x));

    // Polynomial integration
    let poly = expr!(5 * x ^ 4 + 3 * x ^ 2 + 1);
    println!("  ∫ ({poly}) dx = {}", poly.integrate(&x));

    // ── 6. Definite integral ───────────────────────────────────────────
    let zero = symplex::default_context().int(0);
    let one = symplex::default_context().int(1);
    let area = f.definite_integral(&x, &zero, &one);
    println!("\n∫₀¹ f(x) dx = {area}");

    // More definite integrals
    let pi = symplex::default_context().pi();
    let sin_area = x.sin().definite_integral(&x, &zero, &pi);
    println!("∫₀^π sin(x) dx = {}", sin_area.eval());

    let x_squared_area = expr!(x ^ 2).definite_integral(&x, &symplex::default_context().int(-1), &one);
    println!("∫₋₁¹ x² dx = {x_squared_area}");

    // ── 7. Taylor series of sin(x) around 0 ───────────────────────────
    let sin_series = x.sin().maclaurin(&x, 5).unwrap();
    println!("\nsin(x) ≈ {}", sin_series.expand().eval());

    let cos_series = x.cos().maclaurin(&x, 5).unwrap();
    println!("cos(x) ≈ {}", cos_series.expand().eval());

    let exp_series = x.exp().maclaurin(&x, 5).unwrap();
    println!("exp(x) ≈ {}", exp_series.expand().eval());

    // ── 8. Simplification: trig identity ───────────────────────────────
    let trig = expr!(sin(x) ^ 2 + cos(x) ^ 2);
    println!("\n{trig} → {}", trig.simplify());

    // ── 9. More simplification: exp/ln inverse ─────────────────────────
    let exp_ln = x.ln().exp();
    println!("exp(ln(x)) → {}", exp_ln.simplify());

    let ln_exp = x.exp().ln();
    println!("ln(exp(x)) → {}", ln_exp.simplify());

    // Full simplify for complex expressions
    let complicated = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
    println!("(x+1)² - x² - 2x → {}", complicated.full_simplify());

    // ── 10. Limits ─────────────────────────────────────────────────────
    let limit_expr = &x.sin() / &x;
    let lim = limit_expr.limit(&x, &symplex::default_context().int(0)).unwrap();
    println!("\nlim(x→0) sin(x)/x = {lim}");

    // lim(x→0) (exp(x)-1)/x = 1
    let exp_limit = &(&x.exp() - 1) / &x;
    if let Ok(lim2) = exp_limit.limit(&x, &symplex::default_context().int(0)) {
        println!("lim(x→0) (exp(x)-1)/x = {lim2}");
    }

    // Limit at infinity
    let inf = symplex::default_context().infinity();
    if let Ok(lim3) = (1 / &x).limit(&x, &inf) {
        println!("lim(x→∞) 1/x = {lim3}");
    }

    // ── 11. Numerical evaluation ───────────────────────────────────────
    let val = expr!(x ^ 2 + 1).subs_i64(&x, 3);
    println!("\nf(3) where f = x² + 1: {val}");

    let val2 = f.subs_i64(&x, 5);
    println!("f(5) where f = x³ - 3x² + 2x: {val2}");

    // Float evaluation
    let float_val = expr!(sin(x) + cos(x)).eval_f64_with(&[(&x, 1)]).unwrap();
    println!("sin(1) + cos(1) = {float_val:.8}");

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
    let complicated2 = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
    println!("\n(x+1)² - x² - 2x = {}", complicated2.full_simplify());

    // ── 15. Numerical root finding ─────────────────────────────────────
    // Solve x = cos(x) numerically
    let transcendental = &x - &x.cos();
    match transcendental.solve_numeric(&x, 1.0, 50, 1e-12) {
        Ok(root) => println!("\nNumerical root of x - cos(x) = 0: x ≈ {root:.10}"),
        Err(e) => println!("\nNumerical solve failed: {e}"),
    }

    // ── 16. Arbitrary-precision evaluation ─────────────────────────────
    match pi.eval_decimal(30) {
        Ok(s) => println!("\nπ to 30 digits: {s}"),
        Err(e) => println!("\nevalf failed: {e}"),
    }

    // ── 17. ODE Solving ────────────────────────────────────────────────
    println!("\n--- ODE Solving ---");

    vars!(y);

    // Simple separable: y' = x → y = x²/2 + C1
    let dy = y.formal_diff(&x);
    let ode1 = &dy - &x;
    println!("\nODE: y' - x = 0");
    if let Some((sol, constants)) = ode1.solve_ode(&y, &x) {
        println!("  Solution: y = {sol}");
        println!(
            "  Constants: {:?}",
            constants.iter().map(|c| format!("{c}")).collect::<Vec<_>>()
        );
    }

    // Exponential decay: y' + 2y = 0 → y = C1·exp(-2x)
    let ode2 = expr!(diff(y, x) + 2 * y);
    println!("\nODE: y' + 2y = 0");
    if let Some((sol, _)) = ode2.solve_ode(&y, &x) {
        println!("  Solution: y = {sol}");
    }

    // Second-order: y'' + y = 0 → y = C1·cos(x) + C2·sin(x)
    let dy2 = y.formal_diff(&x);
    let d2y2 = dy2.formal_diff(&x);
    let ode3 = &d2y2 + &y;
    println!("\nODE: y'' + y = 0");
    if let Some((sol, constants)) = ode3.solve_ode(&y, &x) {
        println!("  Solution: y = {sol}");
        println!(
            "  Constants: {:?}",
            constants.iter().map(|c| format!("{c}")).collect::<Vec<_>>()
        );
    }

    // y' = 0 → y = C1 (constant)
    let ode_const = y.formal_diff(&x);
    println!("\nODE: y' = 0");
    if let Some((sol, _)) = ode_const.solve_ode(&y, &x) {
        println!("  Solution: y = {sol}");
    }

    // ── 18. ODE Classification ─────────────────────────────────────────
    println!("\n--- ODE Classification ---");

    let ode_sep = expr!(diff(y, x) - x);
    println!("y' - x = 0:    {:?}", ode_sep.classify_ode(&y, &x));

    let ode_lin = expr!(diff(y, x) + 2 * y);
    println!("y' + 2y = 0:   {:?}", ode_lin.classify_ode(&y, &x));

    // ── 19. Verify ODE solutions ───────────────────────────────────────
    println!("\n--- ODE Solution Verification ---");

    // y' - x = 0, solution: y = x²/2
    let ode_check = expr!(diff(y, x) - x);
    let proposed = &x.powi(2) / 2;
    let verified = ode_check.check_ode_solution(&proposed, &y, &x);
    println!("y' = x, proposed y = x²/2: verified = {verified}");

    // y' + 2y = 0, wrong solution: y = x
    let ode_check2 = expr!(diff(y, x) + 2 * y);
    let wrong = x.clone();
    let verified2 = ode_check2.check_ode_solution(&wrong, &y, &x);
    println!("y' + 2y = 0, proposed y = x: verified = {verified2}");

    // ── 20. Implicit differentiation ───────────────────────────────────
    println!("\n--- Implicit Differentiation ---");

    // x² + y² = r² (circle)
    // Differentiate with y depending on x:
    // d/dx(x² + y²) = 2x + 2y·dy/dx
    let circle = expr!(x ^ 2 + y ^ 2);
    let implicit = circle.diff_with_dependent(&x, &[&y]);
    println!("d/dx(x² + y²) with y = y(x):");
    println!("  {implicit}");
    println!("  (This equals 0, so dy/dx = -x/y)");

    // ── 21. Laplace transforms ─────────────────────────────────────────
    println!("\n--- Laplace Transforms ---");

    vars!(t, s);

    // L{1} = 1/s
    if let Ok(result) = symplex::default_context().int(1).laplace(&t, &s) {
        println!("L{{1}} = {result}");
    }

    // L{exp(2t)} = 1/(s-2)
    if let Ok(result) = (&t * 2).exp().laplace(&t, &s) {
        println!("L{{exp(2t)}} = {result}");
    }

    // L{sin(t)} = 1/(s²+1)
    if let Ok(result) = t.sin().laplace(&t, &s) {
        println!("L{{sin(t)}} = {result}");
    }

    // Inverse Laplace: L⁻¹{1/s} = 1
    if let Ok(result) = (1 / &s).inverse_laplace(&s, &t) {
        println!("L⁻¹{{1/s}} = {result}");
    }

    // ── 22. Code generation for derivatives ────────────────────────────
    println!("\n--- Code Generation ---");

    let df_code = df.to_rust_fn("f_prime", &["x"]).unwrap();
    println!("Generated code for f'(x):");
    println!("{df_code}");

    println!("\n=== Done ===");
}
