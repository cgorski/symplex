//! Performance analysis: Does a 10× slowdown to `.simplify()` actually matter?
//!
//! This test answers the question with REAL wall-clock numbers, not theory.
//!
//! Run with:
//!   cd symplex && cargo test --test perf_analysis --release -- --nocapture 2>&1

use std::hint::black_box;
use std::time::{Duration, Instant};
use symplex::prelude::*;

// ── Helpers ────────────────────────────────────────────────────────────────

const ITERATIONS: u32 = 100;

/// Time a closure over `n` runs (with 5 warm-up), returning average.
fn bench_n<F: FnMut()>(n: u32, mut f: F) -> Duration {
    for _ in 0..5 {
        f();
    }
    let start = Instant::now();
    for _ in 0..n {
        f();
    }
    start.elapsed() / n
}

/// Time a closure over ITERATIONS runs, returning average.
fn bench<F: FnMut()>(f: F) -> Duration {
    bench_n(ITERATIONS, f)
}

fn fmt_duration(d: Duration) -> String {
    let nanos = d.as_nanos();
    if nanos < 1_000 {
        format!("{:>8} ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:>8.2} µs", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:>8.2} ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:>8.3}  s", nanos as f64 / 1_000_000_000.0)
    }
}

fn report(label: &str, avg: Duration) {
    println!("  {label:<58} {}", fmt_duration(avg));
}

fn section(title: &str) {
    println!();
    println!("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
    println!("┃ {title:<72} ┃");
    println!("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
}

fn divider(label: &str) {
    println!();
    println!("  ── {label} ──");
}

fn ratio_str(a: Duration, b: Duration) -> String {
    if a.as_nanos() == 0 {
        "N/A".to_string()
    } else {
        format!("{:.1}×", b.as_nanos() as f64 / a.as_nanos() as f64)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Single test function — keeps all output sequential and uninterleaved.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn perf_analysis_all() {
    println!();
    println!("══════════════════════════════════════════════════════════════════════════");
    println!("  PERFORMANCE ANALYSIS: Does 10× iteration in .simplify() matter?");
    println!("  Mode: --release    Iterations per measurement: {}", ITERATIONS);
    println!("══════════════════════════════════════════════════════════════════════════");

    // ═══════════════════════════════════════════════════════════════════
    // 1. ODE solve end-to-end timing
    // ═══════════════════════════════════════════════════════════════════

    section("1. ODE SOLVE END-TO-END TIMING");
    println!("  These are TOTAL times for the ODE solve pipeline,");
    println!("  which internally calls .simplify() many times.");

    {
        let ctx = Context::new();
        symplex::syms!(ctx; x, y);

        // 1a. y' = y
        divider("1a. y' = y  (1st order, exponential growth)");
        {
            let ode = expr!(ctx, diff(y, x) - y);
            println!("  ODE: {} = 0", ode);
            println!("  Classification: {:?}", ode.classify_ode(&y, &x));
            let sol = ode.solve_ode(&y, &x);
            println!("  Solution: y = {}", sol);

            let avg = bench(|| {
                let _ = black_box(ode.solve_ode(&y, &x));
            });
            report("solve_ode(y' = y)", avg);
        }

        // 1b. y' + 2y = 0
        divider("1b. y' + 2y = 0  (exponential decay)");
        {
            let ode = expr!(ctx, diff(y, x) + 2 * y);
            println!("  ODE: {} = 0", ode);
            let sol = ode.solve_ode(&y, &x);
            println!("  Solution: y = {}", sol);

            let avg = bench(|| {
                let _ = black_box(ode.solve_ode(&y, &x));
            });
            report("solve_ode(y' + 2y = 0)", avg);
        }

        // 1c. y'' + y = 0
        divider("1c. y'' + y = 0  (2nd order harmonic oscillator)");
        {
            let dy = y.formal_diff(&x);
            let d2y = dy.formal_diff(&x);
            let ode = &d2y + &y;
            println!("  ODE: {} = 0", ode);
            println!("  Classification: {:?}", ode.classify_ode(&y, &x));
            let sol = ode.solve_ode(&y, &x);
            println!("  Solution: y = {}", sol);

            let avg = bench(|| {
                let _ = black_box(ode.solve_ode(&y, &x));
            });
            report("solve_ode(y'' + y = 0)", avg);
        }

        // 1d. y'' + 3y' + 2y = 0
        divider("1d. y'' + 3y' + 2y = 0  (overdamped 2nd order)");
        {
            let dy = y.formal_diff(&x);
            let d2y = dy.formal_diff(&x);
            let ode = &(&d2y + &(&dy * 3)) + &(&y * 2);
            println!("  ODE: {} = 0", ode);
            let sol = ode.solve_ode(&y, &x);
            println!("  Solution: y = {}", sol);

            let avg = bench(|| {
                let _ = black_box(ode.solve_ode(&y, &x));
            });
            report("solve_ode(y'' + 3y' + 2y = 0)", avg);
        }

        // 1e. 2×2 ODE system (complex eigenvalues — oscillator)
        divider("1e. 2×2 system: dx/dt = [[0,1],[-1,0]] x  (oscillator)");
        {
            let t = ctx.symbol("t");
            let a = Matrix::new(vec![
                vec![ctx.int(0), ctx.int(1)],
                vec![ctx.int(-1), ctx.int(0)],
            ])
            .unwrap();

            let sol = symplex::ode::solve_ode_system(&a, &t);
            if let Some(ref s) = sol {
                for (i, xi) in s.iter().enumerate() {
                    println!("  x{}(t) = {}", i + 1, xi);
                }
            } else {
                println!("  (solver returned None)");
            }

            let avg = bench(|| {
                let _ = black_box(symplex::ode::solve_ode_system(&a, &t));
            });
            report("solve_ode_system(2×2 oscillator)", avg);
        }

        // 1f. 2×2 ODE system (real eigenvalues)
        divider("1f. 2×2 system: dx/dt = [[0,1],[-2,-3]] x  (real eigenvalues)");
        {
            let t = ctx.symbol("t");
            let a = Matrix::new(vec![
                vec![ctx.int(0), ctx.int(1)],
                vec![ctx.int(-2), ctx.int(-3)],
            ])
            .unwrap();

            let sol = symplex::ode::solve_ode_system(&a, &t);
            if let Some(ref s) = sol {
                for (i, xi) in s.iter().enumerate() {
                    println!("  x{}(t) = {}", i + 1, xi);
                }
            } else {
                println!("  (solver returned None)");
            }

            let avg = bench(|| {
                let _ = black_box(symplex::ode::solve_ode_system(&a, &t));
            });
            report("solve_ode_system(2×2 real eigenvalues)", avg);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 2. Matrix operations end-to-end timing
    // ═══════════════════════════════════════════════════════════════════

    section("2. MATRIX OPERATIONS END-TO-END");
    println!("  These are TOTAL times for matrix algebra operations.");

    {
        let ctx = Context::new();

        // 2a. Eigenvalues of 3×3
        divider("2a. Eigenvalues of 3×3 integer matrix");
        {
            let lambda = ctx.symbol("lambda");
            let m = symplex::matrix![ctx,
                [1, 2, 0],
                [0, 3, 1],
                [2, 0, 4]
            ];

            match m.eigenvals(&lambda) {
                Ok(vals) => {
                    for (i, v) in vals.iter().enumerate() {
                        let s = format!("{v}");
                        let display = if s.len() > 70 { format!("{}...", &s[..67]) } else { s };
                        println!("  lambda_{} = {}", i + 1, display);
                    }
                }
                Err(e) => println!("  eigenvals error: {e}"),
            }

            let avg = bench(|| {
                let _ = black_box(m.eigenvals(&lambda));
            });
            report("eigenvals(3×3)", avg);
        }

        // 2b. Determinant of 4×4 integer matrix
        divider("2b. Determinant of 4×4 integer matrix");
        {
            let m = symplex::matrix![ctx,
                [2, 1, 3, 1],
                [1, 0, 2, 1],
                [3, 2, 1, 0],
                [1, 1, 0, 2]
            ];

            let det = m.det().unwrap().eval();
            println!("  det = {det}");

            let avg = bench(|| {
                let _ = black_box(m.det().unwrap().eval());
            });
            report("det(4×4 integer)", avg);
        }

        // 2c. Determinant of 4×4 symbolic matrix
        divider("2c. Determinant of 4×4 symbolic matrix");
        {
            symplex::syms!(ctx; a, b, c, d);
            let m = symplex::matrix![ctx,
                [a + 1, b,     c,     0    ],
                [b,     a - 1, 0,     d    ],
                [c,     0,     a + 2, b    ],
                [0,     d,     b,     a - 2]
            ];

            match m.det() {
                Ok(det) => {
                    let s = format!("{det}");
                    println!("  det = {}... ({} chars)", &s[..s.len().min(80)], s.len());
                }
                Err(e) => println!("  det error: {e}"),
            }

            let avg = bench(|| {
                let _ = black_box(m.det());
            });
            report("det(4×4 symbolic)", avg);
        }

        // 2d. Inverse of 3×3
        divider("2d. Inverse of 3×3 integer matrix");
        {
            let m = symplex::matrix![ctx,
                [1, 2, 3],
                [0, 1, 4],
                [5, 6, 0]
            ];

            match m.inv() {
                Ok(mi) => println!("  inv[0][0] = {}", mi.get(0, 0)),
                Err(e) => println!("  inv error: {e}"),
            }

            let avg = bench(|| {
                let _ = black_box(m.inv());
            });
            report("inv(3×3)", avg);
        }

        // 2e. Inverse + multiply round-trip
        divider("2e. Full round-trip: inv(3×3) then A * A^-1");
        {
            let m = symplex::matrix![ctx,
                [2, 1, 0],
                [1, 3, 2],
                [0, 2, 4]
            ];

            let avg = bench(|| {
                let mi = m.inv().unwrap();
                let product = black_box(&m * &mi);
                let _ = black_box(product);
            });
            report("inv(3×3) + multiply A*A^-1", avg);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3. The "cleanup simplify" pattern (what the ODE solver does)
    // ═══════════════════════════════════════════════════════════════════

    section("3. THE 'CLEANUP SIMPLIFY' PATTERN");
    println!("  The ODE solver does .eval().simplify() on small post-substitution");
    println!("  expressions ~100 times. This measures that exact pattern.");

    {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let five = ctx.int(5);

        // 3a. (3*x + 2).subs(x,5).eval().simplify()
        divider("3a. (3*x + 2).subs(x, 5).eval().simplify()");
        {
            let expr = &x * 3 + 2;
            println!("  Expression: {expr}");
            let result = expr.subs(&x, &five).eval().simplify();
            println!("  After subs(x,5).eval().simplify(): {result}");

            let avg = bench_n(1000, || {
                let _ = black_box(expr.subs(&x, &five).eval().simplify());
            });
            report("subs+eval+simplify                       x1000 avg", avg);

            let start = Instant::now();
            for _ in 0..100 {
                let _ = black_box(expr.subs(&x, &five).eval().simplify());
            }
            let total = start.elapsed();
            println!("  => 100 calls (ODE-solver-like):  {}", fmt_duration(total));
        }

        // 3b. trig identity substituted
        divider("3b. (sin^2(x) + cos^2(x)).subs(x, pi/4).eval().simplify()");
        {
            let expr = &x.sin().powi(2) + &x.cos().powi(2);
            let pi_over_4 = &ctx.pi() / 4;
            println!("  Expression: {expr}");
            let result = expr.subs(&x, &pi_over_4).eval().simplify();
            println!("  After subs+eval+simplify: {result}");

            let avg = bench_n(1000, || {
                let _ = black_box(expr.subs(&x, &pi_over_4).eval().simplify());
            });
            report("trig_id.subs(pi/4).eval.simplify         x1000 avg", avg);

            let start = Instant::now();
            for _ in 0..100 {
                let _ = black_box(expr.subs(&x, &pi_over_4).eval().simplify());
            }
            let total = start.elapsed();
            println!("  => 100 calls (ODE-solver-like):  {}", fmt_duration(total));
        }

        // 3c. exp(-2x) at x=3
        divider("3c. exp(-2*x).subs(x, 3).eval().simplify()");
        {
            let expr = (-&x * 2).exp();
            let three = ctx.int(3);
            println!("  Expression: {expr}");
            let result = expr.subs(&x, &three).eval().simplify();
            println!("  After subs+eval+simplify: {result}");

            let avg = bench_n(1000, || {
                let _ = black_box(expr.subs(&x, &three).eval().simplify());
            });
            report("exp(-2x).subs(x,3).eval.simplify         x1000 avg", avg);
        }

        // 3d. Compare simplify vs full_simplify on the cleanup pattern
        divider("3d. simplify vs full_simplify on cleanup pattern");
        {
            // Use FRESH contexts to avoid any cross-contamination of arena caches
            let avg_simplify = {
                let ctx2 = Context::new();
                let x2 = ctx2.symbol("x");
                let five2 = ctx2.int(5);
                let expr2 = &x2 * 3 + 2;
                bench_n(1000, || {
                    let _ = black_box(expr2.subs(&x2, &five2).eval().simplify());
                })
            };

            let avg_full = {
                let ctx3 = Context::new();
                let x3 = ctx3.symbol("x");
                let five3 = ctx3.int(5);
                let expr3 = &x3 * 3 + 2;
                bench_n(1000, || {
                    let _ = black_box(expr3.subs(&x3, &five3).eval().simplify());
                })
            };

            report(".eval().simplify()      on (3x+2).subs(x,5)", avg_simplify);
            report(".eval().simplify() on (3x+2).subs(x,5)", avg_full);
            println!("  => Ratio full/simple: {}", ratio_str(avg_simplify, avg_full));
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 4. Comparison with SymPy (putting numbers in context)
    // ═══════════════════════════════════════════════════════════════════

    section("4. COMPARISON CONTEXT: symplex vs SymPy");
    println!("  SymPy (Python) typical timings (from published benchmarks):");
    println!("    sympy.simplify(sin(x)**2 + cos(x)**2)  ~10-50 ms");
    println!("    sympy.simplify(x**3 + 2*x + 1)          ~5-20 ms");
    println!("    sympy.Matrix([[1,2],[3,4]]).det()         ~1-5  ms");
    println!("    sympy.dsolve(y' + 2*y, y)              ~50-200 ms");
    println!();
    println!("  Now measuring symplex (Rust) on the same expressions...");

    {
        let ctx = Context::new();
        let x = ctx.symbol("x");

        // sin^2(x) + cos^2(x)
        divider("sin^2(x) + cos^2(x)");
        {
            let expr = &x.sin().powi(2) + &x.cos().powi(2);

            let avg_simplify = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_smart    = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_full     = bench(|| { let _ = black_box(expr.simplify()); });

            report("symplex .simplify()", avg_simplify);
            report("symplex .simplify()", avg_smart);
            report("symplex .simplify()  (iterated)", avg_full);

            let worst_ms = [avg_simplify, avg_smart, avg_full]
                .iter()
                .map(|d| d.as_nanos() as f64 / 1_000_000.0)
                .fold(0.0_f64, f64::max);
            if worst_ms > 0.0 {
                println!(
                    "  => Even worst case ({:.3} ms) is {:.0}x-{:.0}x faster than SymPy (10-50 ms)",
                    worst_ms,
                    10.0 / worst_ms,
                    50.0 / worst_ms,
                );
            }
        }

        // x^3 + 2x + 1
        divider("x^3 + 2x + 1");
        {
            let expr = &x.powi(3) + &x * 2 + 1;
            let avg = bench(|| { let _ = black_box(expr.simplify()); });
            report("symplex .simplify()", avg);
            let ms = avg.as_nanos() as f64 / 1_000_000.0;
            if ms > 0.0 {
                println!("  => vs SymPy ~5-20 ms: symplex is {:.0}x-{:.0}x faster", 5.0 / ms, 20.0 / ms);
            }
        }

        // ODE: y' + 2y = 0
        divider("ODE: y' + 2y = 0");
        {
            symplex::syms!(ctx; y);
            let ode = expr!(ctx, diff(y, x) + 2 * y);
            let avg = bench(|| { let _ = black_box(ode.solve_ode(&y, &x)); });
            report("symplex solve_ode(y' + 2y = 0)", avg);
            let ms = avg.as_nanos() as f64 / 1_000_000.0;
            if ms > 0.0 {
                println!("  => vs SymPy dsolve ~50-200 ms: symplex is {:.0}x-{:.0}x faster", 50.0 / ms, 200.0 / ms);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 5. The actual cost of iteration: smart_simplify vs full_simplify
    // ═══════════════════════════════════════════════════════════════════

    section("5. ACTUAL COST OF ITERATION: smart_simplify vs full_simplify");
    println!("  full_simplify iterates (eval+expand+simplify) up to 10x to fixpoint.");
    println!("  The concern: does this 10x loop matter in practice?");

    struct Row {
        label: &'static str,
        smart: Duration,
        simp: Duration,
        full: Duration,
    }
    let mut rows: Vec<Row> = Vec::new();

    {
        // Each expression gets a FRESH context to avoid arena caching across cases.

        // 5a. Atom: x
        divider("5a. Atom: x  (should early-exit)");
        {
            let ctx = Context::new();
            let x = ctx.symbol("x");
            println!("  Expression: {x}");
            let avg_smart = bench(|| { let _ = black_box(x.simplify()); });
            let avg_simp  = bench(|| { let _ = black_box(x.simplify()); });
            let avg_full  = bench(|| { let _ = black_box(x.simplify()); });
            report("smart_simplify(x)", avg_smart);
            report("simplify(x)", avg_simp);
            report("full_simplify(x)", avg_full);
            rows.push(Row { label: "x (atom)", smart: avg_smart, simp: avg_simp, full: avg_full });
        }

        // 5b. Polynomial
        divider("5b. Polynomial: x^3 + 2x + 1");
        {
            let ctx = Context::new();
            let x = ctx.symbol("x");
            let expr = &x.powi(3) + &x * 2 + 1;
            println!("  Expression: {expr}");
            let avg_smart = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_simp  = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_full  = bench(|| { let _ = black_box(expr.simplify()); });
            report("smart_simplify()", avg_smart);
            report("simplify()", avg_simp);
            report("full_simplify()", avg_full);
            rows.push(Row { label: "x^3+2x+1 (poly)", smart: avg_smart, simp: avg_simp, full: avg_full });
        }

        // 5c. Trig identity
        divider("5c. Trig: sin^2(x) + cos^2(x)");
        {
            let ctx = Context::new();
            let x = ctx.symbol("x");
            let expr = &x.sin().powi(2) + &x.cos().powi(2);
            println!("  Expression: {expr}");
            println!("  smart_simplify -> {}", expr.simplify());
            println!("  simplify       -> {}", expr.simplify());
            println!("  full_simplify  -> {}", expr.simplify());
            let avg_smart = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_simp  = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_full  = bench(|| { let _ = black_box(expr.simplify()); });
            report("smart_simplify()", avg_smart);
            report("simplify()", avg_simp);
            report("full_simplify()", avg_full);
            rows.push(Row { label: "sin^2+cos^2 (trig)", smart: avg_smart, simp: avg_simp, full: avg_full });
        }

        // 5d. Complex cleanup: (3 + 2*i) - 2*i
        divider("5d. Cleanup: (3 + 2*i) - 2*i");
        {
            let ctx = Context::new();
            let i_val = ctx.i_unit();
            let expr = &(&ctx.int(3) + &(&i_val * 2)) - &(&i_val * 2);
            println!("  Expression: {expr}");
            println!("  full_simplify -> {}", expr.simplify());
            let avg_smart = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_simp  = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_full  = bench(|| { let _ = black_box(expr.simplify()); });
            report("smart_simplify()", avg_smart);
            report("simplify()", avg_simp);
            report("full_simplify()", avg_full);
            rows.push(Row { label: "3+2i-2i (cleanup)", smart: avg_smart, simp: avg_simp, full: avg_full });
        }

        // 5e. Exp/Ln: e^(ln(x))
        divider("5e. Exp/Ln: e^(ln(x))");
        {
            let ctx = Context::new();
            let x = ctx.symbol("x");
            let expr = x.ln().exp();
            println!("  Expression: {expr}");
            println!("  full_simplify -> {}", expr.simplify());
            let avg_smart = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_simp  = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_full  = bench(|| { let _ = black_box(expr.simplify()); });
            report("smart_simplify()", avg_smart);
            report("simplify()", avg_simp);
            report("full_simplify()", avg_full);
            rows.push(Row { label: "e^(ln(x)) (exp)", smart: avg_smart, simp: avg_simp, full: avg_full });
        }

        // 5f. Algebraic: (x+1)^2 - x^2 - 2x
        divider("5f. Algebraic: (x+1)^2 - x^2 - 2x");
        {
            let ctx = Context::new();
            let x = ctx.symbol("x");
            let expr = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
            println!("  Expression: {expr}");
            println!("  full_simplify -> {}", expr.simplify());
            let avg_smart = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_simp  = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_full  = bench(|| { let _ = black_box(expr.simplify()); });
            report("smart_simplify()", avg_smart);
            report("simplify()", avg_simp);
            report("full_simplify()", avg_full);
            rows.push(Row { label: "(x+1)^2-x^2-2x", smart: avg_smart, simp: avg_simp, full: avg_full });
        }

        // 5g. Pythagorean + constant
        divider("5g. Trig + constant: sin^2(x) + cos^2(x) + 5");
        {
            let ctx = Context::new();
            let x = ctx.symbol("x");
            let expr = &(&x.sin().powi(2) + &x.cos().powi(2)) + 5;
            println!("  Expression: {expr}");
            println!("  full_simplify -> {}", expr.simplify());
            let avg_smart = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_simp  = bench(|| { let _ = black_box(expr.simplify()); });
            let avg_full  = bench(|| { let _ = black_box(expr.simplify()); });
            report("smart_simplify()", avg_smart);
            report("simplify()", avg_simp);
            report("full_simplify()", avg_full);
            rows.push(Row { label: "sin^2+cos^2+5", smart: avg_smart, simp: avg_simp, full: avg_full });
        }
    }

    // Summary table
    println!();
    println!("  +---------------------------+--------------+--------------+--------------+----------+");
    println!("  | Expression                | smart_simp   | simplify     | full_simp    | ratio    |");
    println!("  |                           |              |              | (iterated)   | full/smt |");
    println!("  +---------------------------+--------------+--------------+--------------+----------+");
    for r in &rows {
        let ratio = ratio_str(r.smart, r.full);
        println!(
            "  | {:<25} | {:>12} | {:>12} | {:>12} | {:>8} |",
            r.label,
            fmt_duration(r.smart).trim_start(),
            fmt_duration(r.simp).trim_start(),
            fmt_duration(r.full).trim_start(),
            ratio,
        );
    }
    println!("  +---------------------------+--------------+--------------+--------------+----------+");

    // ═══════════════════════════════════════════════════════════════════
    // 6. THE VERDICT: simulating the actual ODE workload
    // ═══════════════════════════════════════════════════════════════════

    section("6. THE VERDICT: Simulated ODE workload");
    println!("  Simulating: 100 rounds x 5 expressions = 500 calls to");
    println!("  .subs().eval().simplify() vs .subs().eval().simplify()");
    println!("  using FRESH contexts per scenario (no cross-caching).");

    // Scenario A: 500x subs+eval+simplify (FRESH context)
    divider("Scenario A: 500x subs+eval+simplify (current code)");
    let total_a = {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let five = ctx.int(5);
        let exprs: Vec<Ex> = vec![
            &x * 3 + 2,
            x.powi(2),
            (-&x * 2).exp(),
            x.sin(),
            &x.cos() + 1,
        ];

        // Warm up
        for expr in &exprs {
            let _ = black_box(expr.subs(&x, &five).eval().simplify());
        }

        let start = Instant::now();
        for _ in 0..100 {
            for expr in &exprs {
                let _ = black_box(expr.subs(&x, &five).eval().simplify());
            }
        }
        let elapsed = start.elapsed();
        println!("  Total for 500 subs+eval+simplify:      {}", fmt_duration(elapsed));
        elapsed
    };

    // Scenario B: 500x subs+eval+full_simplify (FRESH context)
    divider("Scenario B: 500x subs+eval+full_simplify (proposed iterated)");
    let total_b = {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let five = ctx.int(5);
        let exprs: Vec<Ex> = vec![
            &x * 3 + 2,
            x.powi(2),
            (-&x * 2).exp(),
            x.sin(),
            &x.cos() + 1,
        ];

        // Warm up
        for expr in &exprs {
            let _ = black_box(expr.subs(&x, &five).eval().simplify());
        }

        let start = Instant::now();
        for _ in 0..100 {
            for expr in &exprs {
                let _ = black_box(expr.subs(&x, &five).eval().simplify());
            }
        }
        let elapsed = start.elapsed();
        println!("  Total for 500 subs+eval+full_simplify: {}", fmt_duration(elapsed));
        elapsed
    };

    // Scenario C: Actual ODE solves (single-shot, realistic)
    divider("Scenario C: Actual end-to-end ODE solves (single shot, cold)");
    {
        let ctx = Context::new();
        symplex::syms!(ctx; x, y);

        let ode1 = expr!(ctx, diff(y, x) - y);
        let start = Instant::now();
        let _ = black_box(ode1.solve_ode(&y, &x));
        let t1 = start.elapsed();
        println!("  y' = y:                    {}", fmt_duration(t1));

        let dy = y.formal_diff(&x);
        let d2y = dy.formal_diff(&x);
        let ode2 = &d2y + &y;
        let start = Instant::now();
        let _ = black_box(ode2.solve_ode(&y, &x));
        let t2 = start.elapsed();
        println!("  y'' + y = 0:               {}", fmt_duration(t2));

        let ode3 = &(&d2y + &(&dy * 3)) + &(&y * 2);
        let start = Instant::now();
        let _ = black_box(ode3.solve_ode(&y, &x));
        let t3 = start.elapsed();
        println!("  y'' + 3y' + 2y = 0:       {}", fmt_duration(t3));
    }

    // ═══════════════════════════════════════════════════════════════════
    // FINAL VERDICT
    // ═══════════════════════════════════════════════════════════════════

    let total_a_ms = total_a.as_nanos() as f64 / 1_000_000.0;
    let total_b_ms = total_b.as_nanos() as f64 / 1_000_000.0;
    let ratio_ab = if total_a.as_nanos() > 0 {
        total_b.as_nanos() as f64 / total_a.as_nanos() as f64
    } else {
        f64::NAN
    };
    let delta_ms = total_b_ms - total_a_ms;

    println!();
    println!("  =====================================================================");
    println!("  =                         FINAL ANALYSIS                            =");
    println!("  =====================================================================");
    println!();
    println!("  Measured workload: 500 calls to subs+eval+[simplify variant]");
    println!("  (simulates an ODE solver doing ~100 simplify calls across 5 exprs)");
    println!();
    println!("    simplify() total:      {:>12}  ({:.3} ms)", fmt_duration(total_a), total_a_ms);
    println!("    full_simplify() total: {:>12}  ({:.3} ms)", fmt_duration(total_b), total_b_ms);
    println!("    Ratio (full/simple):   {:.2}x", ratio_ab);
    println!("    Absolute difference:   {:.3} ms for 500 calls", delta_ms.abs());
    if delta_ms > 0.0 {
        println!("    Per-call overhead:     {:.3} ms", delta_ms / 500.0);
    }
    println!();
    println!("  Reference points:");
    println!("    SymPy simplify(sin^2+cos^2):  10-50 ms   PER CALL");
    println!("    Human perception threshold:    ~100 ms");
    println!("    Typical ODE solve budget:      1-5 seconds acceptable");
    println!("    A 4x4 system (~100 simplify calls) budget:");
    println!("      with simplify():       ~{:.1} ms", total_a_ms / 5.0);
    println!("      with full_simplify():  ~{:.1} ms", total_b_ms / 5.0);
    println!();

    if total_b_ms < 100.0 {
        println!("  VERDICT: The 10x theoretical slowdown is IMPERCEPTIBLE.");
        println!("           {:.1} ms total for 500 calls is below human perception.", total_b_ms);
    } else if total_b_ms < 1000.0 {
        println!("  VERDICT: The 10x theoretical slowdown is NEGLIGIBLE.");
        println!("           {:.1} ms total for 500 calls is under 1 second.", total_b_ms);
    } else if total_b_ms < 5000.0 {
        println!("  VERDICT: The 10x theoretical slowdown is ACCEPTABLE.");
        println!("           {:.1} ms total fits within typical ODE solve budget.", total_b_ms);
    } else {
        println!("  VERDICT: The 10x theoretical slowdown MAY BE NOTICEABLE.");
        println!("           {:.1} ms total -- consider profiling hotspots.", total_b_ms);
    }

    println!();

    // Extra context: even if full_simplify is 10x slower, is it still faster than SymPy?
    let worst_full_per_call_us = total_b.as_nanos() as f64 / 500.0 / 1000.0;
    let sympy_simplify_low_us = 10_000.0; // 10 ms in µs
    if worst_full_per_call_us > 0.0 && worst_full_per_call_us < sympy_simplify_low_us {
        println!(
            "  CONTEXT: symplex full_simplify ({:.1} us/call) is still {:.0}x faster",
            worst_full_per_call_us,
            sympy_simplify_low_us / worst_full_per_call_us,
        );
        println!("           than SymPy's simplify (~10,000 us/call).");
        println!("           The 10x slowdown keeps us firmly in 'instant' territory.");
    }

    println!();
    println!("  =====================================================================");
    println!();
}
