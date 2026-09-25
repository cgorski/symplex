//! Solving in symplex 0.2: solve semantics, linear systems, general
//! (periodic) solutions, polynomial systems, ODE initial-value problems and
//! recurrences.
//!
//! Demonstrates:
//! - `Ex::solve` never lies: identities are `Err(InfiniteSolutions)`,
//!   contradictions and range violations are `Err(NoSolution)`, and results
//!   are evaluated (`asin(1/2)` → `π/6`);
//! - `solve_general` for the full periodic solution families;
//! - `linsolve` / `linsolve_matrix` / `Context::solve_system` with
//!   `LinearSolution::{Unique, Parametric, Inconsistent}` and symbolic
//!   coefficients;
//! - `polysys::solve_system_ex` returning algebraic (radical) solutions;
//! - `solve_numeric_system` (damped Newton);
//! - `solve_ode_ivp`, nth-order constant-coefficient ODEs, Clairaut,
//!   Riccati, and `ode::solve_ode_system_ivp`;
//! - `rsolve_linear` / `rsolve_first_order` for recurrences;
//! - inequalities with absolute values.
//!
//! Run with: `cargo run --example linear_systems_and_ivp`

use symplex::prelude::*;
use symplex::rsolve::{rsolve_first_order, rsolve_linear};

fn show_solution(label: &str, sol: &LinearSolution) {
    match sol {
        LinearSolution::Unique(pairs) => {
            let parts: Vec<String> = pairs
                .iter()
                .map(|(v, val)| format!("{v} = {val}"))
                .collect();
            println!("{label}: unique  {{ {} }}", parts.join(", "));
        }
        LinearSolution::Parametric { solution, free } => {
            let parts: Vec<String> = solution
                .iter()
                .map(|(v, val)| format!("{v} = {val}"))
                .collect();
            let frees: Vec<String> = free.iter().map(|f| f.to_string()).collect();
            println!(
                "{label}: parametric  {{ {} }}   free: {}",
                parts.join(", "),
                frees.join(", ")
            );
        }
        LinearSolution::Inconsistent => println!("{label}: inconsistent (no solution)"),
    }
}

// `x - x` and `0*x + 1` are deliberate: they demonstrate the identity /
// contradiction outcomes of `solve`.
#[allow(clippy::eq_op, clippy::erasing_op)]
fn main() {
    println!("=== Solving: Systems, General Solutions, IVPs, Recurrences ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z, a, b, n, t);

    // ── 1. solve() semantics ────────────────────────────────────────────
    println!("--- Ex::solve semantics ---");
    let half = ctx.rational(1, 2);
    println!(
        "sin(x) = 1/2   → {:?}",
        (&x.sin() - &half).solve(&x).map(strs)
    );
    println!(
        "x² + 1 = 0     → {:?}",
        (&x.powi(2) + 1).solve(&x).map(strs)
    );
    for (label, eq) in [
        ("x − x = 0", &x - &x),
        ("0·x + 1 = 0", &x * 0 + 1),
        ("sin(x) = 2", &x.sin() - 2),
        ("exp(x) = −1", &x.exp() + 1),
        ("|x| = −1", &x.abs() + 1),
    ] {
        match eq.solve(&x) {
            Ok(v) => println!("{label:<14} → Ok({:?})", strs(v)),
            Err(SymplexError::InfiniteSolutions { .. }) => {
                println!("{label:<14} → Err(InfiniteSolutions): identity, every x works")
            }
            Err(SymplexError::NoSolution { reason, .. }) => {
                println!("{label:<14} → Err(NoSolution): {reason}")
            }
            Err(e) => println!("{label:<14} → Err({e})"),
        }
    }

    // ── 2. General (periodic) solutions ─────────────────────────────────
    println!("\n--- solve_general ---");
    let fam = (&x.sin() - &half).solve_general(&x).unwrap();
    println!(
        "sin(x) = 1/2:  {}   with {} ∈ ℤ",
        strs(fam.solutions.clone()).join(",  "),
        fam.parameters[0]
    );
    println!("   instance(k = 1): {}", strs(fam.instance(1)).join(", "));
    let fam = (&(&x * 2).cos() - 1).solve_general(&x).unwrap();
    println!("cos(2x) = 1:   {}", strs(fam.solutions).join(",  "));
    let fam = (&x.tan() - 1).solve_general(&x).unwrap();
    println!("tan(x) = 1:    {}", strs(fam.solutions).join(",  "));
    let fam = (&x.powi(2) - 4).solve_general(&x).unwrap();
    println!(
        "x² = 4:        {}   (non-periodic: {} parameters)",
        strs(fam.solutions).join(", "),
        fam.parameters.len()
    );

    // ── 3. Linear systems ───────────────────────────────────────────────
    println!("\n--- linsolve ---");
    let vars3 = [x.clone(), y.clone(), z.clone()];
    let sol = linsolve(
        &[&x + &y + &z - 6, &x - &y + 2 * &z - 5, &x * 2 + &y - &z - 1],
        &vars3,
    )
    .unwrap();
    show_solution("3×3 determined  ", &sol);
    let sol = linsolve(&[&x + &y + &z - 6, &x - &y - 2], &vars3).unwrap();
    show_solution("2 eqs, 3 unknowns", &sol);
    if let Some(v) = sol.get(&x) {
        println!("   sol.get(x) = {v}");
    }
    let sol = linsolve(&[&x + &y - 1, &x + &y - 2], &[x.clone(), y.clone()]).unwrap();
    show_solution("contradictory   ", &sol);
    // Equations (`eq!`) and symbolic coefficients are fine.
    let sol = linsolve(
        &[eq!(ctx, a * x + y = 1), eq!(ctx, x - y = b)],
        &[x.clone(), y.clone()],
    )
    .unwrap();
    show_solution("symbolic coeffs ", &sol);
    // Matrix form A·x = b (unknowns are named x1, x2, …).
    let am = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let bm = Matrix::col_vector(vec![ctx.int(6), ctx.int(15), ctx.int(24)]).unwrap();
    show_solution("singular A·x = b", &linsolve_matrix(&am, &bm).unwrap());
    // Context::solve_system is the same solver.
    show_solution(
        "ctx.solve_system",
        &ctx.solve_system(&[&x + &y - 3, &x - &y - 1], &[x.clone(), y.clone()])
            .unwrap(),
    );

    // ── 4. Polynomial systems (Gröbner) ─────────────────────────────────
    println!("\n--- polysys::solve_system_ex ---");
    let vars2 = [x.clone(), y.clone()];
    let sols =
        symplex::polysys::solve_system_ex(&[&x.powi(2) + &y.powi(2) - 1, &x - &y], &vars2).unwrap();
    for s in &sols {
        println!("circle ∩ line:      x = {},  y = {}", s[0], s[1]);
    }
    let sols =
        symplex::polysys::solve_system_ex(&[&x.powi(2) + &y.powi(2) - 1, &x.powi(2) - &y], &vars2)
            .unwrap();
    println!(
        "circle ∩ parabola: {} algebraic solutions, e.g. y = {}",
        sols.len(),
        sols[0][1]
    );
    match symplex::polysys::solve_system_ex(&[&x + &y - 1], &vars2) {
        Err(SymplexError::InfiniteSolutions { reason, .. }) => {
            println!("positive-dimensional → Err(InfiniteSolutions): {reason}")
        }
        other => println!("unexpected {other:?}"),
    }

    // ── 5. Numeric systems (damped Newton) ──────────────────────────────
    println!("\n--- solve_numeric_system ---");
    let f1 = &x.powi(2) + &y.powi(2) - 4;
    let f2 = &x.exp() + &y - 1;
    let root = solve_numeric_system(&[f1.clone(), f2.clone()], &vars2, &[1.0, -1.0]).unwrap();
    println!(
        "x² + y² = 4, eˣ + y = 1  →  x ≈ {:.10}, y ≈ {:.10}",
        root[0], root[1]
    );
    let residual = f1.eval_f64_with(&[(&x, root[0]), (&y, root[1])]).unwrap();
    println!("   residual of first equation: {residual:.2e}");

    // ── 6. ODE initial-value problems ───────────────────────────────────
    println!("\n--- ODEs and IVPs ---");
    let yf = ctx.symbol("yf");
    let d = |k: usize| {
        let mut e = yf.clone();
        for _ in 0..k {
            e = e.formal_diff(&x);
        }
        e
    };
    let zero = ctx.int(0);
    // y^(order)(0) = value
    let at0 = |order: usize, value: i64| InitialCondition {
        order,
        x: zero.clone(),
        value: ctx.int(value),
    };

    let ode = &d(2) + &yf; // y'' + y = 0
    let sol = ode.solve_ode_ivp(&yf, &x, &[at0(0, 0), at0(1, 1)]).unwrap();
    println!(
        "y'' + y = 0, y(0)=0, y'(0)=1        → y = {}",
        sol.simplify()
    );

    let ode = &d(1) + &yf * 2; // y' + 2y = 0
    let sol = ode.solve_ode_ivp(&yf, &x, &[at0(0, 3)]).unwrap();
    println!(
        "y' + 2y = 0, y(0)=3                  → y = {}",
        sol.simplify()
    );

    let ode = &d(2) - &d(1) * 3 + &yf * 2 - &x.exp() * 4;
    println!("y'' − 3y' + 2y = 4eˣ  [{:?}]", ode.classify_ode(&yf, &x));
    println!("   general: y = {}", ode.solve_ode(&yf, &x));

    let ode3 = &d(3) - &d(1); // y''' − y' = 0
    println!("y''' − y' = 0  [{:?}]", ode3.classify_ode(&yf, &x));
    println!("   general: y = {}", ode3.solve_ode(&yf, &x));
    let sol = ode3
        .solve_ode_ivp(&yf, &x, &[at0(0, 0), at0(1, 1), at0(2, 0)])
        .unwrap();
    println!("   y(0)=0, y'(0)=1, y''(0)=0: y = {}", sol.simplify());

    let clairaut = &yf - &x * &d(1) - &d(1).powi(2); // y = x y' + (y')²
    println!(
        "y = x·y' + (y')²  [{:?}]        → y = {}",
        clairaut.classify_ode(&yf, &x),
        clairaut.solve_ode(&yf, &x)
    );

    let riccati = &d(1) - &yf.powi(2) + &(&ctx.int(2) / &x.powi(2)); // y' = y² − 2/x²
    println!(
        "y' = y² − 2/x², particular 1/x       → y = {}",
        riccati.solve_riccati(&yf, &x, &(&ctx.int(1) / &x)).unwrap()
    );

    // Linear system x' = A x with x(0) given.
    let a_mat = matrix![ctx, [0, 1], [-1, 0]];
    let sys = symplex::ode::solve_ode_system_ivp(&a_mat, &t, &[ctx.int(1), ctx.int(0)]).unwrap();
    println!(
        "x' = [[0,1],[−1,0]] x, x(0) = (1, 0)  → x(t) = ({}, {})",
        sys[0].simplify(),
        sys[1].simplify()
    );

    // ── 7. Recurrences ──────────────────────────────────────────────────
    println!("\n--- rsolve ---");
    // coeffs are [c₀, c₁, …, c_k] for c₀·a(n) + c₁·a(n+1) + … + c_k·a(n+k) = f(n)
    let fib = rsolve_linear(
        &[ctx.int(-1), ctx.int(-1), ctx.int(1)],
        None,
        &n,
        &[ctx.int(0), ctx.int(1)],
    )
    .unwrap();
    println!("Fibonacci a(n+2) = a(n+1) + a(n):  a(n) = {fib}");
    println!("   a(10) = {}", fib.subs_i64(&n, 10).eval().simplify());
    let hanoi = rsolve_linear(
        &[ctx.int(-2), ctx.int(1)],
        Some(&ctx.int(1)),
        &n,
        &[ctx.int(0)],
    )
    .unwrap();
    println!("Towers of Hanoi a(n+1) = 2a(n) + 1:  a(n) = {hanoi}");
    let general = rsolve_linear(&[ctx.int(6), ctx.int(-5), ctx.int(1)], None, &n, &[]).unwrap();
    println!("a(n+2) − 5a(n+1) + 6a(n) = 0:        a(n) = {general}");
    let tri = rsolve_linear(&[ctx.int(-1), ctx.int(1)], Some(&n), &n, &[ctx.int(0)]).unwrap();
    println!(
        "a(n+1) = a(n) + n, a(0) = 0:         a(n) = {}",
        tri.expand()
    );
    let fact = rsolve_first_order(&(&n + 1), &ctx.int(0), &n, Some(&ctx.int(1))).unwrap();
    println!("a(n+1) = (n+1)·a(n), a(0) = 1:       a(n) = {fact}");

    // ── 8. Inequalities with absolute values ────────────────────────────
    println!("\n--- Inequalities ---");
    println!("|x − 1| < 2  →  {}", (&(&x - 1).abs() - 2).solve_lt(&x));
    println!("|x| ≥ 3      →  {}", (&x.abs() - 3).solve_ge(&x));
    println!("x² − 4 > 0   →  {}", (&x.powi(2) - 4).solve_gt(&x));

    println!("\n✓ Done!");
}

fn strs(v: Vec<Ex>) -> Vec<String> {
    v.iter().map(|e| e.to_string()).collect()
}
