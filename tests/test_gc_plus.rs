//! Integration tests for `liveness_ratio()` and `should_compact()` — arena GC heuristics.

use symplex::prelude::*;

#[test]
fn liveness_ratio_all_live() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2) + &x + 1;
    let ratio = ctx.liveness_ratio(std::slice::from_ref(&expr));
    assert!(ratio > 0.0, "should have some live nodes: {ratio}");
    assert!(ratio <= 1.0, "ratio must be <= 1.0: {ratio}");
}

#[test]
fn liveness_ratio_with_garbage() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Create lots of garbage expressions that won't be rooted
    for i in 0..100 {
        let _ = x.powi(i).sin().cos().exp();
    }
    let keeper = x.clone();
    let ratio = ctx.liveness_ratio(&[keeper]);
    // x alone should be a small fraction of all nodes created
    assert!(ratio < 0.5, "with garbage, liveness should be low: {ratio}");
}

#[test]
fn should_compact_small_arena() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Small arena should never recommend compaction (< 100_000 nodes)
    assert!(
        !ctx.should_compact(&[x]),
        "small arenas should not trigger compaction"
    );
}

#[test]
fn liveness_after_compact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Create some garbage so the original arena has lots of unreachable nodes
    let _ = x.powi(10).expand().sin().cos();
    let expr = &x + 1;

    let ratio_before = ctx.liveness_ratio(std::slice::from_ref(&expr));

    let (new_ctx, new_exprs) = ctx.compact(&[expr]);
    let ratio_after = new_ctx.liveness_ratio(&[new_exprs[0].clone()]);

    // After compact the ratio should improve (or at least not get worse).
    // The new arena only has the transferred subtree plus pre-interned
    // constants, so the ratio won't be 1.0 but must beat the old one.
    assert!(
        ratio_after > ratio_before,
        "compact should improve liveness: before={ratio_before}, after={ratio_after}"
    );
}

#[test]
fn liveness_ratio_empty_roots() {
    let ctx = Context::new();
    let _x = ctx.symbol("x");
    // With no roots, nothing is live — but the function should still not panic
    let ratio = ctx.liveness_ratio(&[]);
    assert!(
        (0.0..=1.0).contains(&ratio),
        "ratio should be in [0, 1]: {ratio}"
    );
    // With no roots, zero nodes are reachable, so ratio should be 0
    assert!(
        (ratio - 0.0).abs() < f64::EPSILON,
        "with no roots, ratio should be 0.0: {ratio}"
    );
}

#[test]
fn liveness_ratio_multiple_roots_increases_liveness() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let expr_x = x.sin();
    let expr_y = y.cos();

    // Liveness with just one root
    let ratio_one = ctx.liveness_ratio(std::slice::from_ref(&expr_x));
    // Liveness with both roots
    let ratio_both = ctx.liveness_ratio(&[expr_x, expr_y]);

    assert!(
        ratio_both >= ratio_one,
        "more roots should mean equal or higher liveness: one={ratio_one}, both={ratio_both}"
    );
}

#[test]
fn should_compact_respects_threshold() {
    // Even with garbage, if the arena is small, should_compact returns false
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for i in 0..200 {
        let _ = x.powi(i).sin().cos().exp().ln();
    }
    // Arena is still well under 100_000 nodes
    assert!(
        !ctx.should_compact(&[x]),
        "should_compact should return false for arenas under 100K nodes (node_count = {})",
        ctx.node_count()
    );
}

#[test]
fn last_compact_size_set_after_compact() {
    // After compacting, a second compact into a third context should reflect
    // the reduced arena size. We verify indirectly: after compact the new arena
    // is small, so should_compact on it should be false.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    let (new_ctx, new_exprs) = ctx.compact(&[expr]);
    assert!(
        !new_ctx.should_compact(&[new_exprs[0].clone()]),
        "freshly compacted context should not need another compaction"
    );
}
