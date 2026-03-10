//! Calculus coverage probe — limits, series, ODE solving.
//! Run with: cargo run --example probe_calculus
//!
//! This is a developer diagnostic tool, NOT a user-facing example.
//! It tests feature coverage and finds regressions in calculus operations.

use symplex::prelude::*;


fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let dy = y.formal_diff(&x);
    let ddy = dy.formal_diff(&x);
    let zero = ctx.int(0);
    let inf = ctx.infinity();

    let mut ode_pass = 0u32;
    let mut ode_fail = 0u32;
    let mut lim_pass = 0u32;
    let mut lim_fail = 0u32;
    let mut ser_pass = 0u32;
    let mut ser_fail = 0u32;

    println!("=== Calculus Coverage Probe ===\n");

    // ═══════════════════════════════════════════════════════════════
    // ODE Solving
    // ═══════════════════════════════════════════════════════════════
    println!("--- ODE Solving ---");

    let ode_cases: Vec<(&str, Ex)> = vec![
        // First-order
        ("y' - x = 0", &dy - &x),
        ("y' - x*y = 0 (separable)", &dy - &(&x * &y)),
        ("y' + 2y = 0 (decay)", &dy + &(&ctx.int(2) * &y)),
        ("y' + 2y = exp(-x)", &dy + &(&ctx.int(2) * &y) - &(-&x).exp()),
        ("y' = 0 (trivial)", dy.clone()),
        // Second-order homogeneous
        ("y'' + y = 0", &ddy + &y),
        ("y'' - 4y = 0", &ddy - &(&ctx.int(4) * &y)),
        ("y'' + 4y' + 4y = 0", &(&ddy + &(&ctx.int(4) * &dy)) + &(&ctx.int(4) * &y)),
        // Second-order nonhomogeneous
        ("y'' + y = sin(x)", &(&ddy + &y) - &x.sin()),
        ("y'' + y = exp(x)", &(&ddy + &y) - &x.exp()),
    ];

    for (label, ode) in &ode_cases {
        let cls = ode.classify_ode(&y, &x);
        let sol = ode.solve_ode(&y, &x);
        if !sol.has_unevaluated() {
            let cnames: Vec<String> = sol.free_symbols().iter().map(|c| format!("{c}")).collect();
            println!("  ✅ {label}");
            println!("     class={cls:?}  y = {sol}  constants={cnames:?}");
            ode_pass += 1;
        } else {
            println!("  ❌ {label}");
            println!("     class={cls:?}  UNSOLVABLE");
            ode_fail += 1;
        }
    }

    // ODE classification-only checks (verify classification is correct even
    // if we can't solve yet)
    println!("\n--- ODE Classification (extra) ---");
    let classify_cases: Vec<(&str, Ex)> = vec![
        ("y' + y = 0 (first-order linear CC)", &dy + &y),
        ("y' = x*y (separable)", &dy - &(&x * &y)),
        ("y'' + 3y' + 2y = 0 (overdamped)", &(&ddy + &(&ctx.int(3) * &dy)) + &(&ctx.int(2) * &y)),
    ];
    for (label, ode) in &classify_cases {
        let cls = ode.classify_ode(&y, &x);
        println!("  📋 {label} → {cls:?}");
    }

    // ODE solution verification
    println!("\n--- ODE Solution Verification ---");
    let ode_verify = expr!(ctx, diff(y, x) - x);
    let proposed_good = &x.powi(2) / 2;
    let proposed_bad = x.clone();
    let v1 = ode_verify.check_ode_solution(&proposed_good, &y, &x);
    let v2 = ode_verify.check_ode_solution(&proposed_bad, &y, &x);
    println!("  y' = x, proposed y = x²/2: {} (expect true)", v1);
    println!("  y' = x, proposed y = x:    {} (expect false)", v2);

    let ode_verify2 = &ddy + &y;
    let cos_sol = x.cos();
    let sin_sol = x.sin();
    let exp_sol = x.exp();
    println!("  y'' + y = 0, y = cos(x): {} (expect true)", ode_verify2.check_ode_solution(&cos_sol, &y, &x));
    println!("  y'' + y = 0, y = sin(x): {} (expect true)", ode_verify2.check_ode_solution(&sin_sol, &y, &x));
    println!("  y'' + y = 0, y = exp(x): {} (expect false)", ode_verify2.check_ode_solution(&exp_sol, &y, &x));

    // ═══════════════════════════════════════════════════════════════
    // Limits
    // ═══════════════════════════════════════════════════════════════
    println!("\n--- Limits ---");

    let limit_cases: Vec<(&str, Ex, Ex)> = vec![
        // L'Hôpital 0/0 forms
        ("sin(x)/x → 0", &x.sin() / &x, zero.clone()),
        ("(e^x - 1)/x → 0", &(&x.exp() - 1) / &x, zero.clone()),
        ("(1 - cos(x))/x² → 0", &(&ctx.int(1) - &x.cos()) / &x.powi(2), zero.clone()),
        // ∞·0 forms
        ("x·e^(-x) → ∞", &x * &(-&x).exp(), inf.clone()),
        ("x²·e^(-x) → ∞", &x.powi(2) * &(-&x).exp(), inf.clone()),
        // ∞/∞ forms
        ("ln(x)/x → ∞", &x.ln() / &x, inf.clone()),
        // Squeeze-theorem style
        ("x·sin(1/x) → 0", &x * &(ctx.int(1) / &x).sin(), zero.clone()),
        // Simple substitution
        ("sin(x) → 0", x.sin(), zero.clone()),
        ("cos(x) → 0", x.cos(), zero.clone()),
        ("exp(x) → 0", x.exp(), zero.clone()),
        // Limit at infinity
        ("1/x → ∞", ctx.int(1) / &x, inf.clone()),
        ("1/x² → ∞", ctx.int(1) / &x.powi(2), inf.clone()),
    ];

    for (label, expr, point) in &limit_cases {
        match expr.try_limit(&x, point) {
            Ok(lim) => {
                println!("  ✅ lim {label} = {lim}");
                lim_pass += 1;
            }
            Err(e) => {
                println!("  ❌ lim {label}: {e}");
                lim_fail += 1;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Series Expansions
    // ═══════════════════════════════════════════════════════════════
    println!("\n--- Series (Maclaurin, order=6) ---");

    let series_cases: Vec<(&str, Ex)> = vec![
        ("sin(x)", x.sin()),
        ("cos(x)", x.cos()),
        ("exp(x)", x.exp()),
        ("1/(1-x)", ctx.int(1) / &(ctx.int(1) - &x)),
        ("ln(1+x)", (&x + 1).ln()),
        ("tan(x)", x.tan()),
        ("(1+x)^(1/2)", (&x + 1).sqrt()),
        ("sinh(x)", x.sinh()),
        ("cosh(x)", x.cosh()),
        ("tanh(x)", x.tanh()),
        ("asin(x)", x.asin()),
        ("atan(x)", x.atan()),
    ];

    for (label, expr) in &series_cases {
        match expr.try_maclaurin(&x, 6) {
            Ok(s) => {
                let expanded = s.expand().eval();
                println!("  ✅ {label} = {expanded}");
                ser_pass += 1;
            }
            Err(e) => {
                println!("  ❌ {label}: {e}");
                ser_fail += 1;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Differentiation round-trips (diff then integrate should recover)
    // ═══════════════════════════════════════════════════════════════
    println!("\n--- Differentiation Sanity ---");

    let diff_cases: Vec<(&str, Ex)> = vec![
        ("x^3", x.powi(3)),
        ("sin(x)", x.sin()),
        ("exp(x)", x.exp()),
        ("ln(x)", x.ln()),
        ("x^2*sin(x)", &x.powi(2) * &x.sin()),
    ];

    for (label, expr) in &diff_cases {
        let d = expr.diff(&x);
        println!("  d/dx({label}) = {d}");
    }

    // Higher-order derivatives
    println!("\n--- Higher-Order Derivatives ---");
    let h = x.powi(5);
    for n in 1..=6 {
        let dn = h.diff_n(&x, n);
        println!("  d^{n}/dx^{n}(x^5) = {dn}");
    }

    // ═══════════════════════════════════════════════════════════════
    // Implicit differentiation
    // ═══════════════════════════════════════════════════════════════
    println!("\n--- Implicit Differentiation ---");
    let circle = &x.powi(2) + &y.powi(2);
    let implicit = circle.diff_with_dependent(&x, &[&y]);
    println!("  d/dx(x² + y²) with y=y(x): {implicit}");

    // ═══════════════════════════════════════════════════════════════
    // Definite integrals
    // ═══════════════════════════════════════════════════════════════
    println!("\n--- Definite Integrals ---");
    let one = ctx.int(1);
    let pi = ctx.pi();

    let def_cases: Vec<(&str, Ex, Ex, Ex)> = vec![
        ("∫₀¹ x² dx", x.powi(2), zero.clone(), one.clone()),
        ("∫₀¹ x³ dx", x.powi(3), zero.clone(), one.clone()),
        ("∫₀^π sin(x) dx", x.sin(), zero.clone(), pi.clone()),
        ("∫₋₁¹ x² dx", x.powi(2), ctx.int(-1), one.clone()),
    ];

    for (label, expr, lo, hi) in &def_cases {
        let result = expr.definite_integral(&x, lo, hi);
        let evaled = result.eval();
        println!("  {label} = {evaled}");
    }

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!("\n=== Summary ===");
    println!("  ODE solving:  {ode_pass} passed, {ode_fail} failed");
    println!("  Limits:       {lim_pass} passed, {lim_fail} failed");
    println!("  Series:       {ser_pass} passed, {ser_fail} failed");

    let total_pass = ode_pass + lim_pass + ser_pass;
    let total_fail = ode_fail + lim_fail + ser_fail;
    println!("  TOTAL:        {total_pass} passed, {total_fail} failed, {} total", total_pass + total_fail);
}
