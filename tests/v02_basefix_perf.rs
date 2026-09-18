//! symplex 0.2 base-layer fixes — construction cost must scale with the
//! DAG, not the unfolded tree.
//!
//! `e ← sin(e) + cos(e)` doubles the tree at every step while the hash-consed
//! DAG only grows by three nodes.  Canonical sort keys used to be the full
//! concatenation of the children's keys (2^depth bytes), which made step 23
//! take tens of seconds.  Keys are now bounded.

use std::time::{Duration, Instant};
use symplex::prelude::*;

/// Wall-clock hang guard.  Two seconds on a developer machine; scaled up on
/// shared CI runners (`CI` is set), which are several times slower and noisy.
fn time_budget(secs: u64) -> std::time::Duration {
    let mult = if std::env::var_os("CI").is_some() {
        5
    } else {
        1
    };
    std::time::Duration::from_secs(secs * mult)
}

fn doubling(depth: usize) -> (Ex, Duration) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut e = x;
    let start = Instant::now();
    for _ in 0..depth {
        e = e.sin() + e.cos();
    }
    (e, start.elapsed())
}

#[test]
fn sin_cos_doubling_depth_30_is_fast() {
    let (e, elapsed) = doubling(30);
    assert!(
        elapsed < time_budget(2),
        "depth-30 doubling took {elapsed:?}"
    );
    // Still a well-formed expression: two summands, both function nodes.
    assert_eq!(e.args().len(), 2);
}

#[test]
fn doubling_ordering_is_deterministic_across_contexts() {
    // The bounded key must still give the same canonical order for the
    // same structure regardless of arena allocation history.
    let (a, _) = doubling(12);
    let ctx = Context::new();
    // Warm the second context with unrelated nodes so ExprIds differ.
    let y = ctx.symbol("y");
    let _junk: Vec<Ex> = (0..50).map(|k| (&y + k).exp()).collect();
    let x = ctx.symbol("x");
    let mut b = x;
    for _ in 0..12 {
        b = b.sin() + b.cos();
    }
    assert_eq!(a.to_string(), b.to_string());
}

#[test]
fn deep_sums_of_wide_products_stay_fast() {
    // Another tree-vs-DAG blow-up shape: p ← p*q + q*p with shared q.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let start = Instant::now();
    let mut p = &x + &y;
    let mut q = &x - &y;
    for _ in 0..25 {
        let np = &p * &q + &q * &p + 1;
        let nq = &p * &p - &q * &q;
        p = np;
        q = nq;
    }
    let elapsed = start.elapsed();
    assert!(elapsed < time_budget(2), "took {elapsed:?}");
    // (Do not display `p`: its unfolded tree, and hence the string, has
    // ~2^25 nodes.  Structural queries stay cheap.)
    // p*q + q*p + 1 combines to 2*p*q + 1.
    assert_eq!(p.args().len(), 2);
    assert!(!p.is_constant());
}
