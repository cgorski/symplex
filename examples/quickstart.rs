//! Symplex Quick Start — a tour of symbolic math in Rust.
//!
//! This example walks through the core capabilities of symplex:
//! expression building, differentiation, compile-time dimensional
//! analysis with physical units, integration, simplification,
//! physical constants, equation solving, typed calculus, matrix
//! algebra, unit conversions, LaTeX output, and code generation.
//!
//! Run with: cargo run --example quickstart

use symplex::prelude::*;


fn main() {
    println!("=== Symplex Quick Start ===\n");

    let __ctx = Context::new();
    symplex::syms!(__ctx; x, y);

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

    // ── 3. Physics with Units ────────────────────────────────────────
    // symplex has compile-time dimensional analysis — the compiler catches
    // unit errors like adding Mass to Length.
    {
        use symplex::units::*;

        let m = Mass::symbol("m");
        let a = Acceleration::symbol("a");

        // Mass × Acceleration → Force (compile-time verified!)
        let f = symplex::dim!(Force: m * a);
        println!("\n--- Physics with Units ---");
        println!("F = m·a = {}", f);

        // This would be a compile error:
        // let bad = &m + &a;  // ERROR: expected Mass, found Acceleration

        // Build complex formulas with expr!, wrap with from_ex
        let __ctx = Context::new();
        symplex::syms!(__ctx; k, x_var);
        let pe = Energy::from_ex(expr!(1/2 * k * x_var^2));
        println!("PE = ½kx² = {}", pe);

        // Derive force from potential energy
        let spring_force = Force::from_ex(-pe.diff(&x_var));
        println!("F = -dPE/dx = {} (Hooke's law!)", spring_force);
        println!();
    }

    // ── 4. Integration ─────────────────────────────────────────────
    println!("--- Integration ---");
    let anti = expr!(x ^ 2).integrate(&x);
    println!("∫ x² dx = {anti}");

    let poly_anti = f.integrate(&x);
    println!("∫ f(x) dx = {poly_anti}");

    // Definite integral
    let zero = __ctx.int(0);
    let one = __ctx.int(1);
    let area = expr!(x ^ 2).definite_integral(&x, &zero, &one);
    println!("∫₀¹ x² dx = {area}");

    // ── 5. Simplification ──────────────────────────────────────────
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

    // ── 6. Physical Constants ────────────────────────────────────────
    // Constants display as symbols (c, h, k_B) but evaluate to exact values.
    {
        use symplex::units::*;
        use symplex::units::constants;

        let c = constants::speed_of_light();  // returns Velocity
        let m = Mass::symbol("m");
        let energy = symplex::dim!(Energy: m * c * c);  // E = mc²

        // Displays symbolically, not as a huge number:
        println!("\n--- Physical Constants ---");
        println!("E = mc² = {}", energy);  // "c^2*m [J]", not "89875517873681764*m"

        // Evaluates to exact value:
        let val = energy.subs(&m, &__ctx.int(1)).eval_f64().unwrap();
        println!("E(m=1kg) = {:.3e} J", val);
        println!();
    }

    // ── 7. Factoring ───────────────────────────────────────────────
    println!("--- Factoring ---");
    println!("x² - 1 = {}", expr!(x ^ 2 - 1).factor(&x));
    println!("x² - 5x + 6 = {}", expr!(x ^ 2 - 5 * x + 6).factor(&x));
    println!("x⁴ - 1 = {}", expr!(x ^ 4 - 1).factor(&x));

    // ── 8. Equation Solving ────────────────────────────────────────
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

    // ── 9. Typed Calculus ─────────────────────────────────────────────
    // DiffWrt: the compiler verifies that d(Length)/d(Time) = Velocity.
    {
        use symplex::units::*;
        let __ctx = Context::new();
        symplex::syms!(__ctx; a, t);
        let t_var = Time::symbol("t");

        let position = Length::from_ex(expr!(1/2 * a * t^2));
        let velocity: Velocity = position.diff_wrt(&t_var);
        let acceleration: Acceleration = velocity.diff_wrt(&t_var);

        println!("\n--- Typed Calculus ---");
        println!("x(t) = {}", position);
        println!("v(t) = dx/dt = {}", velocity);
        println!("a(t) = dv/dt = {}", acceleration);
        println!();
    }

    // ── 10. Numerical Evaluation ────────────────────────────────────
    println!("--- Numerical Evaluation ---");
    let val = expr!(sin(x) + cos(x)).eval_f64_with(&[(&x, 1)]).unwrap();
    println!("sin(1) + cos(1) = {val:.6}");

    let val2 = f.subs_i64(&x, 3);
    println!("f(3) = {val2}");

    // Arbitrary precision
    let pi = __ctx.pi();
    if let Ok(s) = pi.eval_decimal(30) {
        println!("π to 30 digits: {s}");
    }

    // ── 11. Matrix Algebra ──────────────────────────────────────────
    println!("\n--- Matrix Algebra ---");
    let m = matrix![[2, 1], [1, 3]];
    println!("M = {m}");
    println!("det(M) = {}", m.det().unwrap());
    println!("trace(M) = {}", m.trace().unwrap());

    if let Ok(inv) = m.inv() {
        println!("M⁻¹ = {inv}");
    }

    let eigenvals = m.eigenvals(&x).unwrap();
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
    println!("det(B) = {}", sym_m.det().unwrap());

    // ── 12. Unit Conversions ──────────────────────────────────────────
    // Exact rational conversions — no floating-point approximation.
    {
        use symplex::units::*;
        let val = __ctx.int(1);
        println!("\n--- Unit Conversions ---");
        println!("1 hp = {} W (exact!)", Power::horsepower(&val).eval());
        println!("1 psi = {} Pa", Pressure::psi(&val).eval());
        println!("1 atm = {} Pa", Pressure::atmospheres(&val).eval());
        println!("1 nautical mile = {} m", Length::nautical_miles(&val).eval());
        println!();
    }

    // ── 13. LaTeX Output ────────────────────────────────────────────
    println!("--- LaTeX Output ---");
    println!("f(x):    {}", f.to_latex());
    println!("f'(x):   {}", df.to_latex());
    println!("sin²(x): {}", expr!(sin(x) ^ 2).to_latex());
    println!("Matrix:  {}", m.to_latex());

    // ── 14. Limits ─────────────────────────────────────────────────
    println!("\n--- Limits ---");
    let limit_expr = &x.sin() / &x;
    let lim = limit_expr.limit(&x, &__ctx.int(0));
    println!("lim(x→0) sin(x)/x = {lim}");

    // ── 15. Series Expansion ───────────────────────────────────────
    println!("\n--- Series Expansion ---");
    let sin_series = x.sin().maclaurin(&x, 5);
    println!("sin(x) ≈ {}", sin_series.expand().eval());

    // ── 16. Code Generation ────────────────────────────────────────
    println!("\n--- Code Generation ---");
    let code = df.to_rust_fn("f_prime", &["x"]).unwrap();
    println!("Generated Rust function:");
    println!("{code}");

    // ── 17. Compiled Function ──────────────────────────────────────
    println!("--- Compiled Evaluation ---");
    if let Some(compiled) = df.compile(&["x"]) {
        for val in [0.0, 1.0, 2.0, 3.0] {
            println!("  f'({val}) = {:.4}", compiled(&[val]));
        }
    }

    println!("\n✓ Done! See the tutorial at docs/tutorial/ for much more.");
}
