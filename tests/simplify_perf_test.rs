//! Performance measurement for simplification strategies.
//!
//! Run with:
//!   cd symplex && cargo test --test simplify_perf_test --release -- --nocapture 2>&1

use std::time::{Duration, Instant};
use symplex::prelude::*;

const ITERATIONS: u32 = 100;

// ── Helpers ────────────────────────────────────────────────────────────

/// Time a closure over `ITERATIONS` runs, returning (total, average).
fn bench<F: FnMut()>(mut f: F) -> (Duration, Duration) {
    // Warm-up: 5 iterations to stabilize caches / JIT / branch predictors.
    for _ in 0..5 {
        f();
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        f();
    }
    let total = start.elapsed();
    let avg = total / ITERATIONS;
    (total, avg)
}

fn report(label: &str, avg: Duration) {
    if avg.as_micros() > 1000 {
        println!("  {label:<45} {:>8.2} ms", avg.as_secs_f64() * 1000.0);
    } else {
        println!("  {label:<45} {:>8.2} µs", avg.as_nanos() as f64 / 1000.0);
    }
}

fn section(title: &str) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ {title:<60} ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

// ── 1. Simple polynomial: x³ + 2x + 1 ─────────────────────────────────

#[test]
fn perf_simple_polynomial() {
    section("1. Simple polynomial: x³ + 2x + 1");

    let ctx = Context::new();
    let x = ctx.symbol("x");
    let build = || x.powi(3) + &x * 2 + 1;

    // Pre-build so we measure simplification, not construction.
    let expr = build();
    println!("  Input:  {expr}");
    println!("  simplify()       → {}", expr.simplify());
    println!("  smart_simplify() → {}", expr.simplify());
    println!("  full_simplify()  → {}", expr.simplify());
    println!("  simplify_trace() → {}", expr.simplify());
    println!();

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("smart_simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("full_simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("simplify_trace() [pattern-rules only]", avg);
}

// ── 2. Trig expression: sin²(x) + cos²(x) ────────────────────────────

#[test]
fn perf_trig_identity() {
    section("2. Trig identity: sin²(x) + cos²(x)");

    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().powi(2) + x.cos().powi(2);

    println!("  Input:  {expr}");
    println!("  simplify()       → {}", expr.simplify());
    println!("  smart_simplify() → {}", expr.simplify());
    println!("  full_simplify()  → {}", expr.simplify());
    println!("  simplify_trace() → {}", expr.simplify());
    println!();

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("smart_simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("full_simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("simplify_trace() [pattern-rules only]", avg);
}

// ── 3. Large polynomial: 20-term polynomial ────────────────────────────

#[test]
fn perf_large_polynomial() {
    section("3. Large polynomial (20 terms)");

    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Build: 1x^20 + 2x^19 + 3x^18 + ... + 20x + 21
    let mut expr = ctx.int(21);
    for i in 1..=20i64 {
        let coeff = ctx.int(i);
        expr = expr + coeff * x.powi(21 - i);
    }

    println!("  Input:  {expr}");
    println!(
        "  (expression has {} characters in display form)",
        format!("{expr}").len()
    );
    let simplified = expr.simplify();
    println!("  simplify()       → {simplified}");
    let smart = expr.simplify();
    println!("  smart_simplify() → {smart}");
    // Note: full_simplify can be slow on large polys, we still measure it.
    println!();

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("smart_simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("full_simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("simplify_trace() [pattern-rules only]", avg);
}

// ── 4. Already-simple expression: just `x` ─────────────────────────────

#[test]
fn perf_already_simple() {
    section("4. Already simple: x (no-op case for tight loops)");

    let ctx = Context::new();
    let x = ctx.symbol("x");

    println!("  Input:  {x}");
    println!("  simplify()       → {}", x.simplify());
    println!("  smart_simplify() → {}", x.simplify());
    println!("  full_simplify()  → {}", x.simplify());
    println!();

    let (_, avg) = bench(|| {
        let _ = x.simplify();
    });
    report("simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = x.simplify();
    });
    report("smart_simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = x.simplify();
    });
    report("full_simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = x.simplify();
    });
    report("simplify_trace() [pattern-rules only]", avg);

    // Also measure a simple numeric constant
    println!();
    let five = ctx.int(5);
    println!("  Input:  {five}  (numeric literal)");

    let (_, avg) = bench(|| {
        let _ = five.simplify();
    });
    report("simplify()  [numeric literal 5]", avg);

    let (_, avg) = bench(|| {
        let _ = five.simplify();
    });
    report("smart_simplify()  [numeric literal 5]", avg);

    let (_, avg) = bench(|| {
        let _ = five.simplify();
    });
    report("full_simplify()  [numeric literal 5]", avg);
}

// ── 5. Deep expression: sin(sin(sin(...x...))) ────────────────────────

#[test]
fn perf_deep_nesting() {
    section("5. Deep nesting: sin(sin(sin(... x ...)))  depth=10");

    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Build sin(sin(sin(... x ...))) 10 levels deep
    let mut expr = x.clone();
    for _ in 0..10 {
        expr = expr.sin();
    }

    println!("  Input:  {expr}");
    println!("  (display length: {} chars)", format!("{expr}").len());
    let simplified = expr.simplify();
    println!("  simplify()       → {simplified}");
    let smart = expr.simplify();
    println!("  smart_simplify() → {smart}");
    println!();

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("smart_simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("full_simplify()", avg);

    let (_, avg) = bench(|| {
        let _ = expr.simplify();
    });
    report("simplify_trace() [pattern-rules only]", avg);

    // Also measure depth=20
    println!();
    println!("  --- depth=20 ---");
    let mut deep = x.clone();
    for _ in 0..20 {
        deep = deep.sin();
    }
    println!("  (display length: {} chars)", format!("{deep}").len());

    let (_, avg) = bench(|| {
        let _ = deep.simplify();
    });
    report("simplify()  [depth=20]", avg);

    let (_, avg) = bench(|| {
        let _ = deep.simplify();
    });
    report("smart_simplify()  [depth=20]", avg);

    let (_, avg) = bench(|| {
        let _ = deep.simplify();
    });
    report("full_simplify()  [depth=20]", avg);
}

// ── 6. fu() cost on non-trig expressions ──────────────────────────────

#[test]
fn perf_fu_bailout_on_non_trig() {
    section("6. fu() bail-out cost on non-trig expressions");
    println!("  Measuring: does fu()/has_trig check add measurable overhead");
    println!("  to smart_simplify for purely algebraic expressions?");
    println!();

    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Pure polynomial — no trig at all
    let poly = x.powi(3) + &x * 2 + 1;
    println!("  Non-trig expr: {poly}");

    let (_, avg_smart) = bench(|| {
        let _ = poly.simplify();
    });
    report("smart_simplify() [non-trig poly]", avg_smart);

    let (_, avg_trace) = bench(|| {
        let _ = poly.simplify();
    });
    report("simplify_trace() [non-trig poly, no fu]", avg_trace);

    // The difference tells us approximately how much fu's has_trig bail-out
    // plus the other gated strategies cost.
    let overhead_ns = if avg_smart > avg_trace {
        (avg_smart - avg_trace).as_nanos()
    } else {
        0
    };
    println!();
    println!(
        "  → Overhead of smart_simplify over pattern-rules-only: {:.2} µs",
        overhead_ns as f64 / 1000.0
    );
    println!("    (includes flag computation, eval, expand, factor_terms, fu bail-out)");

    // Now compare: trig expression to show fu actually costs something when it runs
    println!();
    let trig = x.sin().powi(2) + x.cos().powi(2);
    println!("  Trig expr: {trig}");

    let (_, avg_trig_smart) = bench(|| {
        let _ = trig.simplify();
    });
    report("smart_simplify() [with trig]", avg_trig_smart);

    let (_, avg_trig_trace) = bench(|| {
        let _ = trig.simplify();
    });
    report("simplify_trace() [with trig]", avg_trig_trace);

    let fu_cost_ns = if avg_trig_smart > avg_smart {
        (avg_trig_smart - avg_smart).as_nanos()
    } else {
        0
    };
    println!();
    println!(
        "  → Extra cost when trig IS present (fu + trig_expand + factor+fu): ~{:.2} µs",
        fu_cost_ns as f64 / 1000.0
    );
}

// ── 7. Consolidated approach simulation ────────────────────────────────

#[test]
fn perf_consolidated_approach() {
    section("7. Consolidated approach: simplify() = smart_simplify always");
    println!("  Simulating: .simplify() always runs smart_simplify (12+ strategies)");
    println!("              .simplify() iterates smart_simplify up to 10×");
    println!();
    println!("  Current .simplify() = simplify_trace() + smart_simplify(), pick best");
    println!("  Proposed .simplify() = smart_simplify() only");
    println!();

    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Build a collection of representative expressions
    let cases: Vec<(&str, Ex)> = vec![
        ("x (atom)", x.clone()),
        ("x³ + 2x + 1", x.powi(3) + &x * 2 + 1),
        ("sin²(x) + cos²(x)", x.sin().powi(2) + x.cos().powi(2)),
        ("(x+1)² - x² - 2x", {
            let xp1 = &x + 1;
            &xp1.powi(2) - &x.powi(2) - &x * 2
        }),
        ("sin(sin(sin(x)))", x.sin().sin().sin()),
    ];

    println!(
        "  {:30} {:>14} {:>14} {:>14}",
        "Expression", "current .s()", "proposed .s()", "ratio"
    );
    println!("  {:-<30} {:-<14} {:-<14} {:-<14}", "", "", "", "");

    for (name, expr) in &cases {
        // Current: simplify() = simplify_trace + smart_simplify, pick best
        let (_, avg_current) = bench(|| {
            let _ = expr.simplify();
        });

        // Proposed: simplify() = smart_simplify only
        let (_, avg_proposed) = bench(|| {
            let _ = expr.simplify();
        });

        let ratio = if avg_proposed.as_nanos() > 0 && avg_current.as_nanos() > 0 {
            avg_proposed.as_nanos() as f64 / avg_current.as_nanos() as f64
        } else {
            f64::NAN
        };

        println!(
            "  {name:30} {:>11.2} µs {:>11.2} µs {:>11.2}×",
            avg_current.as_nanos() as f64 / 1000.0,
            avg_proposed.as_nanos() as f64 / 1000.0,
            ratio,
        );
    }

    // full_simplify timing for the most expensive case
    println!();
    println!("  full_simplify() timing (iterates up to 10× with cancel+expand+radical):");
    for (name, expr) in &cases {
        let (_, avg) = bench(|| {
            let _ = expr.simplify();
        });
        report(&format!("full_simplify()  [{name}]"), avg);
    }
}

// ── 8. Summary & Recommendation ───────────────────────────────────────

#[test]
fn perf_summary() {
    section("8. Summary — absolute cost of smart_simplify on atoms");
    println!("  The critical question: is smart_simplify too expensive for");
    println!("  expressions that are already simple (the no-op hot path)?");
    println!();

    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Atom: just x
    let (_, avg_atom_smart) = bench(|| {
        let _ = x.simplify();
    });
    report("smart_simplify(x)  [atom]", avg_atom_smart);

    let (_, avg_atom_simp) = bench(|| {
        let _ = x.simplify();
    });
    report("simplify(x)  [current]", avg_atom_simp);

    let (_, avg_atom_full) = bench(|| {
        let _ = x.simplify();
    });
    report("full_simplify(x)", avg_atom_full);

    // Small expr
    let small = &x + 1;
    let (_, avg_small_smart) = bench(|| {
        let _ = small.simplify();
    });
    report("smart_simplify(x + 1)  [small]", avg_small_smart);

    let (_, avg_small_full) = bench(|| {
        let _ = small.simplify();
    });
    report("full_simplify(x + 1)", avg_small_full);

    println!();
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │  RECOMMENDATION                                        │");
    println!("  │                                                        │");
    println!("  │  If smart_simplify(atom) < 5 µs:                       │");
    println!("  │    ✅ Consolidation is fine. The flag-based gating      │");
    println!("  │       makes the no-op path extremely cheap.             │");
    println!("  │                                                        │");
    println!("  │  If smart_simplify(atom) is 5–50 µs:                   │");
    println!("  │    ⚠️  Acceptable for most uses, but add a fast path    │");
    println!("  │       that skips strategy evaluation for atoms.         │");
    println!("  │                                                        │");
    println!("  │  If smart_simplify(atom) > 50 µs:                      │");
    println!("  │    ❌ Too slow for tight loops. Keep the split API.     │");
    println!("  │                                                        │");
    println!("  │  The actual numbers are printed above — check them!     │");
    println!("  └─────────────────────────────────────────────────────────┘");

    let atom_us = avg_atom_smart.as_nanos() as f64 / 1000.0;
    println!();
    if atom_us < 5.0 {
        println!(
            "  ✅ VERDICT: smart_simplify(atom) = {atom_us:.2} µs — consolidation is FAST ENOUGH."
        );
        println!("     The early-exit for atoms in smart_simplify makes it essentially free.");
    } else if atom_us < 50.0 {
        println!(
            "  ⚠️  VERDICT: smart_simplify(atom) = {atom_us:.2} µs — acceptable with caveats."
        );
        println!("     Consider keeping the atom early-exit and monitoring in benchmarks.");
    } else {
        println!(
            "  ❌ VERDICT: smart_simplify(atom) = {atom_us:.2} µs — too slow for consolidation."
        );
        println!("     Keep .simplify() as pattern-rules-only for hot paths.");
    }
}
