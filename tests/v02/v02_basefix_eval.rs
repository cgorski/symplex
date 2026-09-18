//! symplex 0.2 base-layer fixes — `eval()` semantics for `Piecewise`.
//!
//! Branches are examined in order.  A condition that evaluates to `False`
//! is dropped, `True` selects that branch (and drops the rest), and an
//! *undecided* condition stops the search: the result is a `Piecewise` of
//! the remaining branches.  A later `True` branch must never shadow an
//! earlier undecided one.

use symplex::prelude::*;

fn s<T: std::fmt::Display>(e: &T) -> String {
    format!("{e}")
}

fn tru(ctx: &Context) -> BoolEx {
    ctx.int(5).gt(&ctx.int(3)).eval()
}

fn fals(ctx: &Context) -> BoolEx {
    ctx.int(3).gt(&ctx.int(5)).eval()
}

#[test]
fn piecewise_undecided_first_branch_is_kept() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let x2 = x.powi(2);
    let two_minus_x = &ctx.int(2) - &x;
    let cond = x.lt(&one);
    let t = tru(&ctx);
    let pw = Ex::piecewise(&[(&x2, &cond), (&two_minus_x, &t)]);
    let ev = pw.eval();
    assert_ne!(
        ev, two_minus_x,
        "later True branch must not shadow x<1: {ev}"
    );
    assert_eq!(ev, pw, "nothing to evaluate — structure preserved: {ev}");
    assert_eq!(
        ev,
        pw.piecewise_simplify(),
        "agrees with piecewise_simplify"
    );
}

#[test]
fn piecewise_decided_first_branch_collapses() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = &x + 1;
    let b = &x + 2;
    let t = tru(&ctx);
    let cond2 = x.gt(&ctx.int(0));
    let pw = Ex::piecewise(&[(&a, &t), (&b, &cond2)]);
    assert_eq!(pw.eval(), a);
    assert_eq!(pw.piecewise_simplify(), a);
}

#[test]
fn piecewise_false_branches_are_dropped_before_undecided() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = fals(&ctx);
    let t = tru(&ctx);
    let cond = x.gt(&ctx.int(0));
    let a = ctx.int(10);
    let b = &x * 3;
    let c = ctx.int(7);
    let pw = Ex::piecewise(&[(&a, &f), (&b, &cond), (&c, &t)]);
    let ev = pw.eval();
    let expected = Ex::piecewise(&[(&b, &cond), (&c, &t)]);
    assert_eq!(ev, expected, "{ev}");
    assert_eq!(ev, pw.piecewise_simplify());
}

#[test]
fn piecewise_conditions_that_evaluate_to_false_are_dropped() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // `3 > 5` is undecided at construction time but evaluates to False.
    let f = ctx.int(3).gt(&ctx.int(5));
    let t = ctx.int(5).gt(&ctx.int(3));
    let a = ctx.int(10);
    let b = &x * 3;
    let pw = Ex::piecewise(&[(&a, &f), (&b, &t)]);
    assert_eq!(pw.eval(), b);
    assert_eq!(pw.piecewise_simplify(), b);
}

#[test]
fn piecewise_all_false_is_nan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = fals(&ctx);
    let pw = Ex::piecewise(&[(&x, &f), (&ctx.int(1), &f)]);
    assert_eq!(s(&pw.eval()), "nan");
    assert_eq!(pw.eval(), pw.piecewise_simplify());
}

#[test]
fn piecewise_true_drops_trailing_branches() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.gt(&ctx.int(0));
    let t = tru(&ctx);
    let a = &x + 1;
    let b = ctx.int(2);
    let c = ctx.int(3);
    let pw = Ex::piecewise(&[(&a, &cond), (&b, &t), (&c, &cond)]);
    let ev = pw.eval();
    let expected = Ex::piecewise(&[(&a, &cond), (&b, &t)]);
    assert_eq!(ev, expected, "{ev}");
    assert_eq!(ev, pw.piecewise_simplify());
}

#[test]
fn piecewise_values_and_conditions_are_evaluated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.gt(&ctx.int(0));
    let t = tru(&ctx);
    let v1 = ctx.pi().sin() + &x; // sin(π) → 0
    let v2 = ctx.int(4).sqrt(); // √4 → 2
    let pw = Ex::piecewise(&[(&v1, &cond), (&v2, &t)]);
    let ev = pw.eval();
    let expected = Ex::piecewise(&[(&x, &cond), (&ctx.int(2), &t)]);
    assert_eq!(ev, expected, "{ev}");
}

#[test]
fn piecewise_nested_inner_collapses_outer_stays() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let t = tru(&ctx);
    let cond = x.gt(&ctx.int(0));
    let inner = Ex::piecewise(&[(&x, &t), (&ctx.int(0), &cond)]);
    let outer = Ex::piecewise(&[(&inner, &cond), (&ctx.int(-1), &t)]);
    let ev = outer.eval();
    let expected = Ex::piecewise(&[(&x, &cond), (&ctx.int(-1), &t)]);
    assert_eq!(ev, expected, "{ev}");
    assert_eq!(ev, outer.piecewise_simplify());
}

#[test]
fn piecewise_nested_in_arithmetic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.gt(&ctx.int(0));
    let t = tru(&ctx);
    let pw = Ex::piecewise(&[(&x, &cond), (&ctx.int(2), &t)]);
    let expr = &pw * 3 + 1;
    let ev = expr.eval();
    assert_eq!(
        ev, expr,
        "undecided piecewise inside a sum is preserved: {ev}"
    );
    // After substitution the branch is decided and the whole thing folds.
    let sub = expr.subs_i64(&x, 5).eval();
    assert_eq!(s(&sub), "16");
    let sub = expr.subs_i64(&x, -5).eval();
    assert_eq!(s(&sub), "7");
}

#[test]
fn piecewise_numeric_evaluation_refuses_undecided_condition() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let t = tru(&ctx);
    let cond = x.gt(&ctx.int(0));
    // Value branches are constant but the first condition is undecided —
    // evalf must not fall through to the `True` branch.
    let pw = Ex::piecewise(&[(&ctx.int(1), &cond), (&ctx.int(2), &t)]);
    assert!(
        pw.eval_f64().is_err(),
        "must not silently pick the else branch"
    );
    // Once decided, evaluation succeeds.
    let v = pw.subs_i64(&x, 3).eval_f64().unwrap();
    assert_eq!(v, 1.0);
    let v = pw.subs_i64(&x, -3).eval_f64().unwrap();
    assert_eq!(v, 2.0);
}
