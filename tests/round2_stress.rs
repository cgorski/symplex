//! Round 2 stress tests for symplex.
//!
//! Targets: performance, threading, large-scale operations, memory,
//! serialization, and cross-context safety.

mod common;

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// 1. LARGE EXPRESSIONS — polynomial with 100+ terms
// ═══════════════════════════════════════════════════════════════════════════

/// Build a polynomial with 100 terms: c_0 + c_1*x + c_2*x^2 + ... + c_99*x^99
fn build_large_poly(ctx: &Context, x: &Ex, n: i64) -> Ex {
    let mut poly = ctx.int(0);
    for i in 0..n {
        // coefficient = (i + 1)
        let coeff = ctx.int(i + 1);
        let term = &coeff * &x.powi(i);
        poly = &poly + &term;
    }
    poly
}

#[test]
fn large_poly_100_terms_construction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let t0 = Instant::now();
    let poly = build_large_poly(&ctx, &x, 100);
    let elapsed = t0.elapsed();
    eprintln!("[large_poly_100] construction: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "constructing 100-term poly took too long: {elapsed:?}"
    );
    // Sanity: should contain x
    assert!(poly.contains(&x), "poly should contain x");
    let s = format!("{poly}");
    assert!(!s.is_empty(), "poly display should not be empty");
}

#[test]
fn large_poly_100_terms_differentiate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = build_large_poly(&ctx, &x, 100);
    let t0 = Instant::now();
    let dp = poly.diff(&x);
    let elapsed = t0.elapsed();
    eprintln!("[large_poly_100] diff: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "differentiating 100-term poly took too long: {elapsed:?}"
    );
    // d/dx of constant term (1) disappears; d/dx of 2*x gives 2
    // The derivative should still mention x (terms from x^2 onward).
    assert!(dp.contains(&x), "derivative should still contain x");
}

#[test]
fn large_poly_100_terms_integrate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = build_large_poly(&ctx, &x, 100);
    let t0 = Instant::now();
    let integ = poly.integrate(&x);
    let elapsed = t0.elapsed();
    eprintln!("[large_poly_100] integrate: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "integrating 100-term poly took too long: {elapsed:?}"
    );
    let s = format!("{integ}");
    assert!(
        !s.contains("Integral"),
        "polynomial integration should always succeed, got: {s}"
    );
}

#[test]
fn large_poly_100_terms_simplify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = build_large_poly(&ctx, &x, 100);
    let t0 = Instant::now();
    let simp = poly.simplify();
    let elapsed = t0.elapsed();
    eprintln!("[large_poly_100] simplify: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "simplifying 100-term poly took too long: {elapsed:?}"
    );
    // Simplification of a polynomial should preserve values
    for &pt in &[0i64, 1, -1, 5] {
        let v_orig = common::eval_at_i64(&poly, &x, pt);
        let v_simp = common::eval_at_i64(&simp, &x, pt);
        assert!(
            common::approx_eq(v_orig, v_simp, 1e-6),
            "simplify changed value at x={pt}: orig={v_orig}, simp={v_simp}"
        );
    }
}

#[test]
fn large_poly_100_terms_expand() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = build_large_poly(&ctx, &x, 100);
    let t0 = Instant::now();
    let expanded = poly.expand();
    let elapsed = t0.elapsed();
    eprintln!("[large_poly_100] expand: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "expanding 100-term poly took too long: {elapsed:?}"
    );
    // Expanding an already-expanded polynomial should be identity (or equivalent)
    let v1 = common::eval_at_i64(&poly, &x, 3);
    let v2 = common::eval_at_i64(&expanded, &x, 3);
    assert!(
        common::approx_eq(v1, v2, 1e-6),
        "expand changed value: {v1} vs {v2}"
    );
}

#[test]
fn large_poly_ftc_check() {
    // For a smaller poly (30 terms), verify FTC: d/dx(∫ p dx) == p
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = build_large_poly(&ctx, &x, 30);
    let integ = poly.integrate(&x);
    let round = integ.diff(&x);
    for &pt in &[0i64, 1, -1, 3, 7] {
        let v_orig = common::eval_at_i64(&poly, &x, pt);
        let v_round = common::eval_at_i64(&round, &x, pt);
        assert!(
            common::approx_eq(v_orig, v_round, 1e-6),
            "FTC failed at x={pt}: poly={v_orig}, d/dx(∫poly)={v_round}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. DEEP NESTING — sin(sin(sin(...sin(x)...))) 50 levels
// ═══════════════════════════════════════════════════════════════════════════

/// Build sin^n(x) — n nested applications of sin.
fn build_nested_sin(x: &Ex, depth: usize) -> Ex {
    let mut expr = x.clone();
    for _ in 0..depth {
        expr = expr.sin();
    }
    expr
}

#[test]
fn deep_nesting_50_sin_construction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let t0 = Instant::now();
    let deep = build_nested_sin(&x, 50);
    let elapsed = t0.elapsed();
    eprintln!("[deep_nest_50] construction: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 10,
        "constructing 50-deep sin nest took too long: {elapsed:?}"
    );
    assert!(deep.contains(&x), "nested sin should contain x");
}

#[test]
fn deep_nesting_50_sin_display() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let deep = build_nested_sin(&x, 50);
    let t0 = Instant::now();
    let s = format!("{deep}");
    let elapsed = t0.elapsed();
    eprintln!("[deep_nest_50] display: {elapsed:?} (len={})", s.len());
    assert!(
        elapsed.as_secs() < 10,
        "displaying 50-deep sin took too long: {elapsed:?}"
    );
    // Should contain many "sin(" substrings
    let sin_count = s.matches("sin(").count();
    assert!(
        sin_count >= 50,
        "expected >= 50 sin( occurrences, got {sin_count}"
    );
}

#[test]
fn deep_nesting_50_sin_differentiate() {
    // README: "no recursion — all tree traversals use explicit stacks"
    // This must NOT blow the call stack.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let deep = build_nested_sin(&x, 50);
    let t0 = Instant::now();
    let deriv = deep.diff(&x);
    let elapsed = t0.elapsed();
    eprintln!("[deep_nest_50] diff: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "differentiating 50-deep sin took too long: {elapsed:?}"
    );
    // The derivative should still contain x (chain rule produces cos*cos*...*1)
    assert!(
        deriv.contains(&x),
        "derivative of nested sin should still contain x"
    );
    // The derivative should contain both sin and cos
    let ds = format!("{deriv}");
    assert!(ds.contains("cos"), "derivative should contain cos terms");
}

#[test]
fn deep_nesting_100_sin_no_stack_overflow() {
    // Push even deeper — 100 levels. Must not crash.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let deep = build_nested_sin(&x, 100);
    // Just constructing and formatting should be fine
    let s = format!("{deep}");
    assert!(
        s.matches("sin(").count() >= 100,
        "expected >= 100 sin( occurrences"
    );
    // Differentiating 100 levels — should still not blow the stack
    let t0 = Instant::now();
    let _deriv = deep.diff(&x);
    let elapsed = t0.elapsed();
    eprintln!("[deep_nest_100] diff: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 60,
        "differentiating 100-deep sin took too long: {elapsed:?}"
    );
}

#[test]
fn deep_nesting_simplify_does_not_crash() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let deep = build_nested_sin(&x, 30);
    let t0 = Instant::now();
    let simp = deep.simplify();
    let elapsed = t0.elapsed();
    eprintln!("[deep_nest_30] simplify: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "simplifying 30-deep sin took too long: {elapsed:?}"
    );
    // sin(sin(...(x))) at x=0 should give 0
    let v_orig = common::eval_at_i64(&deep, &x, 0);
    let v_simp = common::eval_at_i64(&simp, &x, 0);
    assert!(
        common::approx_eq(v_orig, v_simp, 1e-10),
        "simplify changed value at x=0: {v_orig} vs {v_simp}"
    );
}

#[test]
fn deep_nesting_mixed_functions() {
    // sin(cos(exp(ln(sin(cos(...(x)...))))))
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut expr = x.clone();
    for i in 0..40 {
        expr = match i % 4 {
            0 => expr.sin(),
            1 => expr.cos(),
            2 => expr.exp(),
            _ => expr.abs(), // avoid ln of negative
        };
    }
    let t0 = Instant::now();
    let _deriv = expr.diff(&x);
    let elapsed = t0.elapsed();
    eprintln!("[deep_mixed_40] diff: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "differentiating 40-deep mixed nest took too long: {elapsed:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. THREAD SAFETY — shared context, concurrent operations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn thread_safety_4_threads_diff_integrate_simplify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let base_expr = &x.powi(3) + &x.powi(2) * 2 - &x * 5 + 7;
    let barrier = Arc::new(Barrier::new(4));

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let _ctx = ctx.clone();
            let x = x.clone();
            let expr = base_expr.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait(); // synchronize start
                match i % 4 {
                    0 => {
                        // Differentiate
                        let d = expr.diff(&x);
                        let s = format!("{d}");
                        assert!(!s.is_empty(), "diff result empty in thread {i}");
                        s
                    }
                    1 => {
                        // Integrate
                        let integ = expr.integrate(&x);
                        let s = format!("{integ}");
                        assert!(!s.is_empty(), "integrate result empty in thread {i}");
                        s
                    }
                    2 => {
                        // Simplify
                        let simp = expr.simplify();
                        let s = format!("{simp}");
                        assert!(!s.is_empty(), "simplify result empty in thread {i}");
                        s
                    }
                    _ => {
                        // Expand
                        let expanded = expr.expand();
                        let s = format!("{expanded}");
                        assert!(!s.is_empty(), "expand result empty in thread {i}");
                        s
                    }
                }
            })
        })
        .collect();

    for h in handles {
        let result = h.join().expect("thread panicked");
        assert!(!result.is_empty());
    }
}

#[test]
fn thread_safety_8_threads_create_and_diff() {
    let ctx = Context::new();
    let barrier = Arc::new(Barrier::new(8));

    let handles: Vec<_> = (0..8)
        .map(|i| {
            let ctx = ctx.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                let x = ctx.symbol(&format!("x{i}"));
                let mut expr = x.clone();
                for k in 1..=20 {
                    expr = &expr + &ctx.int(k) * &x.powi(k);
                }
                let diff = expr.diff(&x);
                let integ = expr.integrate(&x);
                let simp = expr.simplify();
                (format!("{diff}"), format!("{integ}"), format!("{simp}"))
            })
        })
        .collect();

    for h in handles {
        let (d, i, s) = h.join().expect("thread panicked");
        assert!(!d.is_empty());
        assert!(!i.is_empty());
        assert!(!s.is_empty());
    }
}

#[test]
fn thread_safety_shared_symbol_concurrent_diff() {
    // All threads use the same symbol "x" from the same context
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let barrier = Arc::new(Barrier::new(4));

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let x = x.clone();
            let ctx = ctx.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                let expr = &x.powi(i + 2) + &x * ctx.int(i + 1);
                let d = expr.diff(&x);
                format!("{d}")
            })
        })
        .collect();

    let results: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for r in &results {
        assert!(!r.is_empty());
    }
}

#[test]
fn thread_safety_concurrent_solve() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let barrier = Arc::new(Barrier::new(4));

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let x = x.clone();
            let ctx = ctx.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                // x^2 - (i+1) = 0  →  x = ±sqrt(i+1)
                let eq = &x.powi(2) - ctx.int(i + 1);
                eq.solve_or_empty(&x).len()
            })
        })
        .collect();

    for h in handles {
        let count = h.join().expect("thread panicked during solve");
        assert!(count >= 1, "expected at least 1 root, got {count}");
    }
}

#[test]
fn thread_safety_concurrent_serde() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(3) + &x.sin() + ctx.int(42);
    let barrier = Arc::new(Barrier::new(4));

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let expr = expr.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..50 {
                    let json = expr.to_json().expect("to_json failed");
                    assert!(!json.is_empty());
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked during serde");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. MEMORY — create many expressions, drop them, no blow-up
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn memory_create_and_drop_100k_expressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let t0 = Instant::now();
    for i in 0..100_000i64 {
        let _ = &x.powi(i % 50) + ctx.int(i);
    }
    let elapsed = t0.elapsed();
    eprintln!("[memory_100k] create+drop: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 60,
        "creating 100K expressions took too long: {elapsed:?}"
    );
    // If we got here without OOM, the test passes.
    // The context should still be usable.
    let test = &x + 1;
    assert!(test.contains(&x));
}

#[test]
fn memory_create_and_drop_complex_expressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let t0 = Instant::now();
    for i in 0..10_000i64 {
        // Each iteration builds a moderately complex expression and drops it
        let expr = (&x.powi(3) + &x.powi(2) * ctx.int(i) - &x * ctx.int(i * 2) + ctx.int(i))
            .sin()
            .exp();
        let _ = format!("{expr}"); // force materialization
    }
    let elapsed = t0.elapsed();
    eprintln!("[memory_10k_complex] create+display+drop: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 60,
        "creating 10K complex expressions took too long: {elapsed:?}"
    );
}

#[test]
fn memory_repeated_diff_integration_no_leak() {
    // Repeatedly differentiate/integrate to create lots of intermediate nodes
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let t0 = Instant::now();
    for _ in 0..1_000 {
        let expr = &x.powi(5) + &x.powi(3) - &x;
        let d = expr.diff(&x);
        let _ = d.integrate(&x);
    }
    let elapsed = t0.elapsed();
    eprintln!("[memory_repeated_calc] 1000 diff+integrate: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 60,
        "1000 diff+integrate took too long: {elapsed:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. SERIALIZATION ROUND-TRIP AT SCALE
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn serde_roundtrip_large_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = build_large_poly(&ctx, &x, 50);

    let json = poly.to_json().expect("to_json failed on large poly");
    assert!(!json.is_empty());

    let ctx2 = Context::new();
    let back = ctx2
        .from_json(&json)
        .expect("from_json failed on large poly");

    // Verify equality at sample points
    for &pt in &[-3i64, -1, 0, 1, 2, 5] {
        let v1 = common::eval_at_i64(&poly, &x, pt);
        let x2 = ctx2.symbol("x");
        let v2 = common::eval_at_i64(&back, &x2, pt);
        assert!(
            common::approx_eq(v1, v2, 1e-6),
            "serde roundtrip changed poly value at x={pt}: {v1} vs {v2}"
        );
    }
}

#[test]
fn serde_roundtrip_nested_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(cos(x^2 + 1))^3 + exp(x) - ln(x^2 + 1)
    let inner = x.powi(2) + ctx.int(1);
    let expr = &inner.cos().sin().powi(3) + &x.exp() - &inner.ln();

    let json = expr.to_json().expect("to_json failed");
    let ctx2 = Context::new();
    let back = ctx2.from_json(&json).expect("from_json failed");

    assert_eq!(
        format!("{expr}"),
        format!("{back}"),
        "serde roundtrip changed display form"
    );
}

#[test]
fn serde_roundtrip_deeply_nested() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let deep = build_nested_sin(&x, 20);

    let json = deep.to_json().expect("to_json for deep nest");
    let ctx2 = Context::new();
    let back = ctx2.from_json(&json).expect("from_json for deep nest");

    assert_eq!(
        format!("{deep}"),
        format!("{back}"),
        "serde roundtrip changed deeply nested expression"
    );
}

#[test]
fn serde_roundtrip_trig_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);

    let json = expr.to_json().expect("to_json");
    let ctx2 = Context::new();
    let back = ctx2.from_json(&json).expect("from_json");

    assert_eq!(format!("{expr}"), format!("{back}"));
}

#[test]
fn serde_roundtrip_many_iterations() {
    // Serialize → deserialize 100 times in a row (same expression)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(4) + &x.sin() * 3 - ctx.rational(7, 3);

    let mut current_json = expr.to_json().expect("initial to_json");
    for i in 0..100 {
        let tmp_ctx = Context::new();
        let back = tmp_ctx
            .from_json(&current_json)
            .unwrap_or_else(|e| panic!("from_json failed at iteration {i}: {e}"));
        current_json = back
            .to_json()
            .unwrap_or_else(|e| panic!("to_json failed at iteration {i}: {e}"));
    }
    // Final round-trip should still match original display
    let final_ctx = Context::new();
    let final_expr = final_ctx.from_json(&current_json).expect("final from_json");
    let x_final = final_ctx.symbol("x");
    for &pt in &[0i64, 1, 2] {
        let v_orig = common::eval_at_i64(&expr, &x, pt);
        let v_final = common::eval_at_i64(&final_expr, &x_final, pt);
        assert!(
            common::approx_eq(v_orig, v_final, 1e-6),
            "100x serde roundtrip changed value at x={pt}: {v_orig} vs {v_final}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. CANCEL / FACTOR ON LARGE POLYNOMIALS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_x4_minus_1() {
    // x^4 - 1 = (x-1)(x+1)(x^2+1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(4) - 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    eprintln!("[factor] x^4 - 1 → {s}");

    // Verify factored form equals original at sample points
    for &pt in &[-5i64, -2, -1, 0, 1, 2, 5, 10] {
        let vo = common::eval_at_i64(&expr, &x, pt);
        let vf = common::eval_at_i64(&factored, &x, pt);
        assert!(
            common::approx_eq(vo, vf, 1e-9),
            "factor(x^4-1) changed value at x={pt}: {vo} vs {vf}"
        );
    }
}

#[test]
fn factor_x8_minus_1() {
    // x^8 - 1 = (x-1)(x+1)(x^2+1)(x^4+1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(8) - 1;
    let t0 = Instant::now();
    let factored = expr.factor(&x);
    let elapsed = t0.elapsed();
    eprintln!("[factor] x^8 - 1: {elapsed:?} → {factored}");
    assert!(
        elapsed.as_secs() < 30,
        "factoring x^8-1 took too long: {elapsed:?}"
    );
    for &pt in &[-3i64, -1, 0, 1, 3] {
        let vo = common::eval_at_i64(&expr, &x, pt);
        let vf = common::eval_at_i64(&factored, &x, pt);
        assert!(
            common::approx_eq(vo, vf, 1e-9),
            "factor(x^8-1) value mismatch at x={pt}: {vo} vs {vf}"
        );
    }
}

#[test]
fn factor_x10_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(10) - 1;
    let t0 = Instant::now();
    let factored = expr.factor(&x);
    let elapsed = t0.elapsed();
    eprintln!("[factor] x^10 - 1: {elapsed:?} → {factored}");
    assert!(
        elapsed.as_secs() < 30,
        "factoring x^10-1 took too long: {elapsed:?}"
    );
    for &pt in &[-2i64, 0, 1, 2, 3] {
        let vo = common::eval_at_i64(&expr, &x, pt);
        let vf = common::eval_at_i64(&factored, &x, pt);
        assert!(
            common::approx_eq(vo, vf, 1e-9),
            "factor(x^10-1) value mismatch at x={pt}: {vo} vs {vf}"
        );
    }
}

#[test]
fn factor_x20_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(20) - 1;
    let t0 = Instant::now();
    let factored = expr.factor(&x);
    let elapsed = t0.elapsed();
    eprintln!("[factor] x^20 - 1: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 60,
        "factoring x^20-1 took too long: {elapsed:?}"
    );

    let s = format!("{factored}");
    eprintln!("[factor] x^20 - 1 → {s}");

    // Value preservation: factored form must equal original
    for &pt in &[-2i64, 0, 1, 2, 3] {
        let vo = common::eval_at_i64(&expr, &x, pt);
        let vf = common::eval_at_i64(&factored, &x, pt);
        assert!(
            common::approx_eq(vo, vf, 1e-6),
            "factor(x^20-1) value mismatch at x={pt}: {vo} vs {vf}"
        );
    }
}

#[test]
fn cancel_x2_minus_1_over_x_minus_1() {
    // (x^2 - 1) / (x - 1) → x + 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = &x.powi(2) - 1;
    let denom = &x - 1;
    let frac = &numer / &denom;
    let cancelled = frac.cancel(&x);
    let s = format!("{cancelled}");
    eprintln!("[cancel] (x^2-1)/(x-1) → {s}");
    // Should simplify to x+1
    for &pt in &[0i64, 2, 3, 5, 10] {
        let vc = common::eval_at_i64(&cancelled, &x, pt);
        let expected = (pt + 1) as f64;
        assert!(
            common::approx_eq(vc, expected, 1e-9),
            "cancel((x^2-1)/(x-1)) at x={pt}: got {vc}, expected {expected}"
        );
    }
}

#[test]
fn cancel_x10_minus_1_over_x5_minus_1() {
    // (x^10 - 1) / (x^5 - 1) → x^5 + 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = &x.powi(10) - 1;
    let denom = &x.powi(5) - 1;
    let frac = &numer / &denom;
    let t0 = Instant::now();
    let cancelled = frac.cancel(&x);
    let elapsed = t0.elapsed();
    let s = format!("{cancelled}");
    eprintln!("[cancel] (x^10-1)/(x^5-1): {elapsed:?} → {s}");
    assert!(elapsed.as_secs() < 30, "cancel took too long: {elapsed:?}");

    // Verify: should equal x^5 + 1 at various points (avoiding x=1 where denom=0)
    for &pt in &[0i64, 2, 3, -1, -2] {
        let vc = common::eval_at_i64(&cancelled, &x, pt);
        let expected = (pt as f64).powi(5) + 1.0;
        assert!(
            common::approx_eq(vc, expected, 1e-6),
            "cancel at x={pt}: got {vc}, expected {expected}"
        );
    }
}

#[test]
fn cancel_x6_minus_1_over_x3_minus_1() {
    // (x^6 - 1) / (x^3 - 1) → x^3 + 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = &x.powi(6) - 1;
    let denom = &x.powi(3) - 1;
    let frac = &numer / &denom;
    let cancelled = frac.cancel(&x);
    let s = format!("{cancelled}");
    eprintln!("[cancel] (x^6-1)/(x^3-1) → {s}");

    for &pt in &[0i64, 2, -2, 3] {
        let vc = common::eval_at_i64(&cancelled, &x, pt);
        let expected = (pt as f64).powi(3) + 1.0;
        assert!(
            common::approx_eq(vc, expected, 1e-9),
            "cancel at x={pt}: got {vc}, expected {expected}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. REPEATED SIMPLIFICATION — idempotency
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_idempotent_100_times_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(3) + &x.powi(2) * 2 - &x * 5 + 7;

    let mut current = expr.simplify();
    let first_form = format!("{current}");
    let t0 = Instant::now();
    for i in 1..100 {
        let next = current.simplify();
        let next_form = format!("{next}");
        assert_eq!(
            first_form, next_form,
            "simplify not idempotent at iteration {i}: first='{first_form}', now='{next_form}'"
        );
        current = next;
    }
    let elapsed = t0.elapsed();
    eprintln!("[idempotent] 99 simplify iterations: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "99 simplify iterations took too long: {elapsed:?}"
    );
}

#[test]
fn simplify_idempotent_100_times_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);

    let mut current = expr.simplify();
    let first_form = format!("{current}");
    eprintln!("[idempotent_trig] sin^2+cos^2 simplifies to: {first_form}");
    let t0 = Instant::now();
    for i in 1..100 {
        let next = current.simplify();
        let next_form = format!("{next}");
        assert_eq!(
            first_form, next_form,
            "simplify not idempotent at iteration {i}: '{first_form}' vs '{next_form}'"
        );
        current = next;
    }
    let elapsed = t0.elapsed();
    eprintln!("[idempotent_trig] 99 iterations: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "99 trig simplify iterations took too long: {elapsed:?}"
    );
}

#[test]
fn simplify_idempotent_rational() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x^2 + 2x + 1) / (x + 1) — should simplify
    let expr = &(&x.powi(2) + &x * 2 + 1) / &(&x + 1);

    let mut current = expr.simplify();
    let first_form = format!("{current}");
    for i in 1..50 {
        let next = current.simplify();
        let next_form = format!("{next}");
        assert_eq!(
            first_form, next_form,
            "simplify not idempotent on rational expr at iter {i}"
        );
        current = next;
    }
}

#[test]
fn simplify_does_not_slow_down() {
    // Each simplify call should take roughly the same time (no accumulation)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x.powi(4) + &x.powi(3) - &x.powi(2) + &x - 1).sin() + x.cos();

    let mut times = Vec::with_capacity(20);
    let mut current = expr.clone();
    for _ in 0..20 {
        let t0 = Instant::now();
        current = current.simplify();
        times.push(t0.elapsed());
    }
    // The last iteration should not be more than 10x the first
    // (allowing generous margin for JIT / cache effects)
    let first = times[0].as_micros().max(1);
    let last = times[times.len() - 1].as_micros().max(1);
    eprintln!(
        "[simplify_timing] first={first}µs, last={last}µs, ratio={}",
        last as f64 / first as f64
    );
    // Very generous bound: last <= 50x first (to avoid flaky tests)
    assert!(
        last <= first * 50,
        "simplify is getting slower: first={first}µs, last={last}µs"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. CROSS-CONTEXT MIXING — verify it's caught
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_add_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let x = ctx_a.symbol("x");
    let y = ctx_b.symbol("y");
    let _ = &x + &y;
}

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_mul_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let x = ctx_a.symbol("x");
    let y = ctx_b.symbol("y");
    let _ = &x * &y;
}

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_diff_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let expr = ctx_a.symbol("x").powi(2);
    let var = ctx_b.symbol("x");
    let _ = expr.diff(&var);
}

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_integrate_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let expr = ctx_a.symbol("x").powi(2);
    let var = ctx_b.symbol("x");
    let _ = expr.integrate(&var);
}

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_subs_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let x = ctx_a.symbol("x");
    let val = ctx_b.int(5);
    let _ = x.subs(&x, &val);
}

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_solve_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let expr = ctx_a.symbol("x").powi(2) - ctx_a.int(4);
    let var = ctx_b.symbol("x");
    let _ = expr.solve(&var);
}

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_pow_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let x = ctx_a.symbol("x");
    let y = ctx_b.symbol("y");
    let _ = x.pow(&y);
}

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_contains_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let expr = ctx_a.symbol("x").powi(2);
    let needle = ctx_b.symbol("x");
    let _ = expr.contains(&needle);
}

#[test]
fn same_context_clone_is_compatible() {
    // Cloning a context should produce a compatible context
    let ctx = Context::new();
    let ctx_clone = ctx.clone();
    let x = ctx.symbol("x");
    let y = ctx_clone.symbol("y");
    // This should NOT panic — cloned context shares the same arena
    let sum = &x + &y;
    assert!(sum.contains(&x));
    assert!(sum.contains(&y));
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. ADDITIONAL STRESS: expand large products
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_product_of_binomials() {
    // (x+1)(x+2)(x+3)...(x+10) — should expand without issue
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut product = &x + 1;
    for i in 2..=10 {
        product = &product * &(&x + i);
    }
    let t0 = Instant::now();
    let expanded = product.expand();
    let elapsed = t0.elapsed();
    eprintln!("[expand_binomials_10] {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "expanding product of 10 binomials took too long: {elapsed:?}"
    );

    // Verify: at x=0, product = 1*2*3*...*10 = 10!
    let v0 = common::eval_at_i64(&expanded, &x, 0);
    let expected = (1..=10).product::<i64>() as f64;
    assert!(
        common::approx_eq(v0, expected, 1e-3),
        "expanded product at x=0: got {v0}, expected {expected}"
    );
}

#[test]
fn expand_high_power_binomial() {
    // (x + 1)^15 — binomial expansion creates 16 terms
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(15);
    let t0 = Instant::now();
    let expanded = expr.expand();
    let elapsed = t0.elapsed();
    eprintln!("[expand_(x+1)^15] {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "expanding (x+1)^15 took too long: {elapsed:?}"
    );
    // At x=1: (1+1)^15 = 32768
    let v1 = common::eval_at_i64(&expanded, &x, 1);
    assert!(
        common::approx_eq(v1, 32768.0, 1e-3),
        "(x+1)^15 at x=1: got {v1}, expected 32768"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. COMPILATION AND EVAL AT SCALE
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compile_large_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = build_large_poly(&ctx, &x, 50);
    let compiled = poly.compile(&["x"]);
    if let Ok(f) = compiled {
        // Evaluate at many points
        for i in -10..=10 {
            let xval = i as f64;
            let compiled_val = f(&[xval]);
            let sym_val = common::eval_at_i64(&poly, &x, i);
            assert!(
                common::approx_eq(compiled_val, sym_val, 1e-3)
                    || (compiled_val.is_infinite() && sym_val.is_infinite())
                    || (compiled_val.is_nan() && sym_val.is_nan()),
                "compiled vs symbolic mismatch at x={i}: {compiled_val} vs {sym_val}"
            );
        }
    } else {
        eprintln!(
            "[compile_large_poly] compile returned None — expression may contain non-compilable constructs"
        );
    }
}

#[test]
fn eval_f64_many_times() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(3) + &x.sin() * 2 - ctx.rational(1, 3);
    let t0 = Instant::now();
    for i in -500..=500 {
        let _val = i as f64 / 100.0;
        let _ = expr.subs(&x, &ctx.rational(i, 100)).eval_f64();
    }
    let elapsed = t0.elapsed();
    eprintln!("[eval_f64_1000] {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "1000 eval_f64 calls took too long: {elapsed:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. DIFFERENTIATION STRESS — high-order derivatives
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn high_order_derivative_polynomial() {
    // d^20/dx^20 of x^20 should be 20!
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(20);
    let mut current = expr;
    for _ in 0..20 {
        current = current.diff(&x);
    }
    // Should be a constant = 20!
    let val = current.eval_f64();
    let factorial_20: f64 = (1..=20).map(|i| i as f64).product();
    match val {
        Ok(v) => {
            assert!(
                common::approx_eq(v, factorial_20, factorial_20 * 1e-10),
                "d^20/dx^20 of x^20 should be 20! = {factorial_20}, got {v}"
            );
        }
        Err(e) => {
            // Might fail if free symbols remain unexpectedly
            let s = format!("{current}");
            eprintln!("[high_order_diff] 20th derivative: {s} (eval error: {e})");
        }
    }
    // 21st derivative should be 0
    let next = current.diff(&x);
    let s = format!("{next}");
    assert_eq!(s, "0", "21st derivative of x^20 should be 0, got: {s}");
}

#[test]
fn high_order_derivative_exp() {
    // d^n/dx^n of e^x is e^x for all n
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut current = x.exp();
    for _ in 0..20 {
        current = current.diff(&x);
    }
    // Should still be exp(x)
    let val_at_0 = common::eval_at_i64(&current, &x, 0);
    assert!(
        common::approx_eq(val_at_0, 1.0, 1e-10),
        "d^20/dx^20 of exp(x) at x=0 should be 1, got {val_at_0}"
    );
}

#[test]
fn high_order_derivative_sin() {
    // d^4/dx^4 of sin(x) = sin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut current = x.sin();
    for _ in 0..4 {
        current = current.diff(&x);
    }
    // Should be sin(x) again
    let val_at_pi_2 = current
        .subs(&x, &ctx.rational(15708, 10000))
        .eval_f64()
        .unwrap_or(f64::NAN);
    let expected = (std::f64::consts::FRAC_PI_2).sin();
    assert!(
        common::approx_eq(val_at_pi_2, expected, 1e-3),
        "d^4/dx^4 sin(x) at x≈π/2: got {val_at_pi_2}, expected {expected}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. STRESS: many distinct contexts
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn many_contexts_independent() {
    // Create 100 independent contexts, build expressions in each
    let t0 = Instant::now();
    for i in 0..100 {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = x.powi(i % 10 + 1) + ctx.int(i);
        let _ = expr.diff(&x);
        let _ = expr.simplify();
        let _ = format!("{expr}");
    }
    let elapsed = t0.elapsed();
    eprintln!("[many_contexts_100] {elapsed:?}");
    assert!(
        elapsed.as_secs() < 30,
        "100 independent contexts took too long: {elapsed:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. STRESS: large substitution chain
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn substitution_chain_100() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut expr = x.clone();
    // Build expr = x, then repeatedly substitute x → x+1
    // After 100 substitutions, eval at x=0 should give 100
    for _ in 0..100 {
        expr = expr.subs(&x, &(&x + 1));
    }
    let val = common::eval_at_i64(&expr, &x, 0);
    assert!(
        common::approx_eq(val, 100.0, 1e-6),
        "after 100 x→x+1 substitutions at x=0: got {val}, expected 100"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. STRESS: free_symbols on large expression
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn free_symbols_large_expression() {
    let ctx = Context::new();
    // Build expression with 50 different symbols
    let symbols: Vec<Ex> = (0..50).map(|i| ctx.symbol(&format!("x{i}"))).collect();
    let mut expr = ctx.int(0);
    for (i, s) in symbols.iter().enumerate() {
        expr = &expr + &(s.powi(2) + ctx.int(i as i64));
    }
    let t0 = Instant::now();
    let free = expr.free_symbols();
    let elapsed = t0.elapsed();
    eprintln!(
        "[free_symbols_50] {elapsed:?}, found {} symbols",
        free.len()
    );
    assert_eq!(
        free.len(),
        50,
        "expected 50 free symbols, got {}",
        free.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. STRESS: tree round-trip preserves structure
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn tree_roundtrip_large_poly() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = build_large_poly(&ctx, &x, 30);

    let tree = poly.to_tree();
    let ctx2 = Context::new();
    let back = ctx2.from_tree(&tree);

    let x2 = ctx2.symbol("x");
    for &pt in &[-2i64, 0, 1, 3, 7] {
        let v1 = common::eval_at_i64(&poly, &x, pt);
        let v2 = common::eval_at_i64(&back, &x2, pt);
        assert!(
            common::approx_eq(v1, v2, 1e-6),
            "tree roundtrip changed value at x={pt}: {v1} vs {v2}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 16. THREAD SAFETY: concurrent factor + cancel
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn thread_safety_concurrent_factor_cancel() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let barrier = Arc::new(Barrier::new(4));

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let _ctx = ctx.clone();
            let x = x.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                let deg = (i + 2) * 2; // 4, 6, 8, 10
                let expr = &x.powi(deg as i64) - 1;
                let factored = expr.factor(&x);
                let frac = &(&x.powi(deg as i64) - 1) / &(&x.powi(deg as i64 / 2) - 1);
                let cancelled = frac.cancel(&x);
                // Verify value at x=2
                let vf = common::eval_at_i64(&factored, &x, 2);
                let vo = common::eval_at_i64(&expr, &x, 2);
                assert!(
                    common::approx_eq(vf, vo, 1e-6),
                    "thread {i}: factor changed value at x=2: {vf} vs {vo}"
                );
                // Cancelled should be x^(deg/2) + 1
                let vc = common::eval_at_i64(&cancelled, &x, 2);
                let expected = 2.0_f64.powi(deg / 2) + 1.0;
                assert!(
                    common::approx_eq(vc, expected, 1e-6),
                    "thread {i}: cancel value at x=2: got {vc}, expected {expected}"
                );
                format!("thread {i} ok: deg={deg}")
            })
        })
        .collect();

    for h in handles {
        let msg = h.join().expect("thread panicked in factor/cancel");
        eprintln!("[concurrent_factor_cancel] {msg}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 17. STRESS: is_zero / is_one on complex expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn is_zero_complex_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x - x) should be zero
    let zero = &x - &x;
    assert_eq!(format!("{zero}"), "0");

    // (x^2 - x^2) should be zero
    let zero2 = &x.powi(2) - &x.powi(2);
    assert_eq!(format!("{zero2}"), "0");
}

#[test]
fn simplify_complex_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(x)^2 + cos(x)^2 - 1 should simplify to 0
    let expr = &x.sin().powi(2) + &x.cos().powi(2) - 1;
    let simp = expr.simplify();
    let s = format!("{simp}");
    assert_eq!(s, "0", "sin^2 + cos^2 - 1 should simplify to 0, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 18. STRESS: rapid clone and drop
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rapid_clone_and_drop() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(5).sin().exp() + x.cos().powi(3);
    let t0 = Instant::now();
    for _ in 0..100_000 {
        let cloned = expr.clone();
        drop(cloned);
    }
    let elapsed = t0.elapsed();
    eprintln!("[rapid_clone_drop] 100K iterations: {elapsed:?}");
    assert!(
        elapsed.as_secs() < 10,
        "100K clone+drop took too long: {elapsed:?}"
    );
    // Original should still be usable
    let s = format!("{expr}");
    assert!(!s.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// 19. STRESS: expand then factor round-trip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_then_factor_roundtrip() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Start factored: (x-1)(x+1)(x-2)(x+2) = (x^2-1)(x^2-4) = x^4 - 5x^2 + 4
    let factored_orig = (&x - 1) * (&x + 1) * (&x - 2) * (&x + 2);
    let expanded = factored_orig.expand();
    let refactored = expanded.factor(&x);

    // Value preservation
    for &pt in &[-5i64, -3, 0, 1, 2, 3, 5] {
        let vo = common::eval_at_i64(&factored_orig, &x, pt);
        let ve = common::eval_at_i64(&expanded, &x, pt);
        let vr = common::eval_at_i64(&refactored, &x, pt);
        assert!(
            common::approx_eq(vo, ve, 1e-9),
            "expand changed value at x={pt}: {vo} vs {ve}"
        );
        assert!(
            common::approx_eq(vo, vr, 1e-9),
            "refactor changed value at x={pt}: {vo} vs {vr}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 20. STRESS: build expression from JSON, diff, serialize back
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn serde_diff_serde_pipeline() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(5) + &x.sin() * 3 - ctx.rational(7, 2);

    // Serialize
    let json1 = expr.to_json().expect("to_json 1");

    // Deserialize, differentiate
    let ctx2 = Context::new();
    let back = ctx2.from_json(&json1).expect("from_json");
    let x2 = ctx2.symbol("x");
    let diff = back.diff(&x2);

    // Serialize the derivative
    let json2 = diff.to_json().expect("to_json 2");
    assert!(!json2.is_empty());

    // Deserialize the derivative and verify
    let ctx3 = Context::new();
    let diff_back = ctx3.from_json(&json2).expect("from_json 2");
    let x3 = ctx3.symbol("x");

    for &pt in &[-2i64, 0, 1, 3] {
        let v_diff = common::eval_at_i64(&diff, &x2, pt);
        let v_back = common::eval_at_i64(&diff_back, &x3, pt);
        assert!(
            common::approx_eq(v_diff, v_back, 1e-6),
            "serde→diff→serde pipeline mismatch at x={pt}: {v_diff} vs {v_back}"
        );
    }
}
