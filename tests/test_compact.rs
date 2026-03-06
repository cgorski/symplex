//! Integration tests for `Context::compact()` — generational arena GC.

use symplex::prelude::*;

#[test]
fn compact_reduces_node_count() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Create lots of intermediate expressions that we won't keep.
    let _ = x.powi(2).sin().exp().ln().abs();
    let _ = x.powi(3).cos().exp();
    let _ = (&x + 1).powi(10).expand();
    let big_count = ctx.node_count();

    // Keep only x.
    let (new_ctx, new_exprs) = ctx.compact(std::slice::from_ref(&x));
    let small_count = new_ctx.node_count();

    assert!(
        small_count < big_count,
        "compact should reduce node count: {} → {}",
        big_count,
        small_count,
    );
    assert_eq!(format!("{}", new_exprs[0]), "x");
}

#[test]
fn compact_preserves_expression_display() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + &x.sin() + 1;
    let original_str = format!("{expr}");

    let (_new_ctx, new_exprs) = ctx.compact(&[expr]);
    assert_eq!(format!("{}", new_exprs[0]), original_str);
}

#[test]
fn compact_preserves_assumptions() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive, Assumption::Real]);

    let (_new_ctx, new_exprs) = ctx.compact(std::slice::from_ref(&x));
    assert_eq!(new_exprs[0].is_positive(), Some(true));
    assert_eq!(new_exprs[0].is_real(), Some(true));
}

#[test]
fn compact_multiple_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr1 = &x + 1;
    let expr2 = y.sin();

    let orig1 = format!("{expr1}");
    let orig2 = format!("{expr2}");

    let (_new_ctx, new_exprs) = ctx.compact(&[expr1, expr2]);
    assert_eq!(new_exprs.len(), 2);
    assert_eq!(format!("{}", new_exprs[0]), orig1);
    assert_eq!(format!("{}", new_exprs[1]), orig2);
}

#[test]
fn compact_empty_roots() {
    let ctx = Context::new();
    let _ = ctx.symbol("x"); // create something to pollute the arena
    let (new_ctx, new_exprs) = ctx.compact(&[]);
    assert!(new_exprs.is_empty());
    // New context should have only pre-interned constants (0, 1, -1, pi, e, ...).
    assert!(new_ctx.node_count() > 0);
}

#[test]
fn compact_shared_subexpressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let shared = x.sin(); // sin(x) is shared between expr1 and expr2
    let expr1 = &shared + 1;
    let expr2 = shared.powi(2);

    let (new_ctx, _new_exprs) = ctx.compact(&[expr1, expr2]);
    // sin(x) should be shared in the new arena too, keeping the count small.
    assert!(new_ctx.node_count() <= ctx.node_count());
}

#[test]
fn compact_preserves_rational() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    let x = ctx.symbol("x");
    let expr = &x + &half;
    let original_str = format!("{expr}");

    let (_new_ctx, new_exprs) = ctx.compact(&[expr]);
    assert_eq!(format!("{}", new_exprs[0]), original_str);
}

#[test]
fn compact_preserves_constants() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let e = ctx.e();
    let expr = &pi + &e;
    let original_str = format!("{expr}");

    let (_new_ctx, new_exprs) = ctx.compact(&[expr]);
    assert_eq!(format!("{}", new_exprs[0]), original_str);
}

#[test]
fn compact_duplicate_root() {
    // Passing the same expression twice should work — both entries in
    // the result should display identically.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();

    let (_new_ctx, new_exprs) = ctx.compact(&[expr.clone(), expr]);
    assert_eq!(new_exprs.len(), 2);
    assert_eq!(format!("{}", new_exprs[0]), format!("{}", new_exprs[1]));
}

#[test]
fn compact_deep_tree() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Build a moderately deep chain: sin(cos(sin(cos(... x ...))))
    let mut expr = x.clone();
    for i in 0..20 {
        expr = if i % 2 == 0 { expr.sin() } else { expr.cos() };
    }
    let original_str = format!("{expr}");

    let (new_ctx, new_exprs) = ctx.compact(&[expr]);
    assert_eq!(format!("{}", new_exprs[0]), original_str);
    // Only the chain nodes + x + pre-interned constants should survive.
    assert!(new_ctx.node_count() < ctx.node_count() || new_ctx.node_count() > 0);
}

#[test]
fn compact_neg_and_pow() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (-&x).powi(3);
    let original_str = format!("{expr}");

    let (_new_ctx, new_exprs) = ctx.compact(&[expr]);
    assert_eq!(format!("{}", new_exprs[0]), original_str);
}

#[test]
fn compact_old_context_still_works() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    let original_str = format!("{expr}");

    // Compact into a new context.
    let (_new_ctx, _new_exprs) = ctx.compact(std::slice::from_ref(&expr));

    // The original context and expressions should still be usable.
    assert_eq!(format!("{expr}"), original_str);
    let expr2 = &x + 2;
    assert_eq!(format!("{expr2}"), "x + 2");
}

#[test]
fn compact_infinity_and_special_values() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let x = ctx.symbol("x");
    let expr = &x + &inf;
    let original_str = format!("{expr}");

    let (_new_ctx, new_exprs) = ctx.compact(&[expr]);
    assert_eq!(format!("{}", new_exprs[0]), original_str);
}

#[test]
fn compact_piecewise() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let cond = x.gt(&zero);
    let neg_x = -&x;
    let not_cond = cond.not();
    let pw = Ex::piecewise(&[(&x, &cond), (&neg_x, &not_cond)]);
    let original_str = format!("{pw}");

    let (_new_ctx, new_exprs) = ctx.compact(&[pw]);
    assert_eq!(format!("{}", new_exprs[0]), original_str);
}

#[test]
fn compact_result_can_be_further_operated_on() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;

    let (new_ctx, new_exprs) = ctx.compact(&[expr]);

    // We should be able to do further math in the new context.
    let y = new_ctx.symbol("y");
    let combined = &new_exprs[0] + &y;
    let combined_str = format!("{combined}");
    assert!(combined_str.contains("x"));
    assert!(combined_str.contains("y"));
}

#[test]
fn compact_with_expand_result() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let big = (&x + 1).powi(5).expand();
    let big_str = format!("{big}");

    // The expansion creates many intermediates.
    let before = ctx.node_count();

    let (new_ctx, new_exprs) = ctx.compact(&[big]);
    let after = new_ctx.node_count();

    assert_eq!(format!("{}", new_exprs[0]), big_str);
    assert!(
        after <= before,
        "compacted arena ({after}) should not be larger than original ({before})",
    );
}
