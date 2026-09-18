//! Concurrency tests for symplex.
//!
//! Verifies that the thread-safe `Ex` type (Send + Sync via Arc<RwLock>)
//! works correctly under concurrent access.

use std::thread;
use symplex::prelude::*;

/// Multiple threads constructing expressions on a shared context.
#[test]
fn concurrent_expression_construction() {
    let ctx = Context::new();
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let ctx = ctx.clone();
            thread::spawn(move || {
                let x = ctx.symbol(&format!("x{i}"));
                let mut expr = x.clone();
                for j in 1..=100 {
                    expr = &expr + ctx.int(j);
                }
                expr
            })
        })
        .collect();

    let results: Vec<Ex> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.len(), 8);
    // Each result should contain its own symbol
    for (i, result) in results.iter().enumerate() {
        let sym_name = format!("x{i}");
        let sym = ctx.symbol(&sym_name);
        assert!(
            result.contains(&sym),
            "result {i} should contain {sym_name}"
        );
    }
}

/// Multiple threads reading (Display, is_zero, free_symbols) while
/// one thread constructs.
#[test]
fn concurrent_read_while_write() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + &x + 1;

    let readers: Vec<_> = (0..4)
        .map(|_| {
            let expr = expr.clone();
            let x = x.clone();
            thread::spawn(move || {
                for _ in 0..100 {
                    let s = format!("{expr}");
                    assert!(!s.is_empty());
                    let _ = expr.free_symbols();
                    let _ = expr.contains(&x);
                }
            })
        })
        .collect();

    // Writer thread
    let ctx2 = ctx.clone();
    let writer = thread::spawn(move || {
        for i in 0..100 {
            let _ = ctx2.int(i);
        }
    });

    for r in readers {
        r.join().unwrap();
    }
    writer.join().unwrap();
}

/// Shared context: multiple threads calling ctx.symbol()
/// should all get consistent results.
#[test]
fn global_context_thread_safety() {
    let ctx = Context::new();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let ctx = ctx.clone();
            thread::spawn(move || {
                let x = ctx.symbol("x");
                let y = ctx.symbol("y");
                let expr = &x + &y;
                format!("{expr}")
            })
        })
        .collect();

    let results: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    // All threads should produce the same canonical form
    for r in &results {
        assert_eq!(r, &results[0], "all threads should get same canonical form");
    }
}

/// Simplify and expand called concurrently on the same expression.
#[test]
fn concurrent_simplify_and_expand() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(2);

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let expr = expr.clone();
            thread::spawn(move || {
                if i % 2 == 0 {
                    let expanded = expr.expand();
                    format!("{expanded}")
                } else {
                    let simplified = expr.simplify();
                    format!("{simplified}")
                }
            })
        })
        .collect();

    for h in handles {
        let result = h.join().unwrap();
        assert!(!result.is_empty());
    }
}

/// Diff, integrate, and solve from multiple threads.
#[test]
fn concurrent_calculus_operations() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(3) - &x;

    let h1 = {
        let expr = expr.clone();
        let x = x.clone();
        thread::spawn(move || {
            let d = expr.diff(&x);
            format!("{d}")
        })
    };

    let h2 = {
        let expr = expr.clone();
        let x = x.clone();
        thread::spawn(move || {
            let integ = expr.integrate(&x);
            format!("{integ}")
        })
    };

    let h3 = {
        let expr = expr.clone();
        let x = x.clone();
        thread::spawn(move || {
            let roots = expr.solve_or_empty(&x);
            roots.len()
        })
    };

    let diff_result = h1.join().unwrap();
    let int_result = h2.join().unwrap();
    let solve_count = h3.join().unwrap();

    assert!(!diff_result.is_empty());
    assert!(!int_result.is_empty());
    assert!(
        solve_count >= 2,
        "x^3 - x should have at least 2 roots, got {solve_count}"
    );
}
