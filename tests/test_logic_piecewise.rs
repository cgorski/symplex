//! Tests for boolean expressions, logic, relationals, and piecewise.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Relational construction and display
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gt_display() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.gt(&ctx.int(0));
    assert_eq!(format!("{cond}"), "x > 0");
}

#[test]
fn lt_display() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.lt(&ctx.int(5));
    let s = format!("{cond}");
    // lt(x, 5) is stored as gt(5, x), displayed as "5 > x" or "x < 5"
    assert!(s.contains("x") && s.contains("5"), "lt: {s}");
}

#[test]
fn ge_display() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.ge(&ctx.int(0));
    let s = format!("{cond}");
    assert!(s.contains("x") && s.contains(">="), "ge: {s}");
}

#[test]
fn eq_display() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.eq_expr(&ctx.int(3));
    let s = format!("{cond}");
    assert!(s.contains("==") && s.contains("x"), "eq: {s}");
}

#[test]
fn ne_display() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.ne_expr(&ctx.int(0));
    let s = format!("{cond}");
    assert!(s.contains("!=") && s.contains("x"), "ne: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Relational evaluation (numeric args)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gt_true() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.int(5).gt(&ctx.int(3)).eval()), "True");
}

#[test]
fn gt_false() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.int(2).gt(&ctx.int(7)).eval()), "False");
}

#[test]
fn ge_equal() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.int(3).ge(&ctx.int(3)).eval()), "True");
}

#[test]
fn lt_true() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.int(1).lt(&ctx.int(5)).eval()), "True");
}

#[test]
fn le_true() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.int(3).le(&ctx.int(3)).eval()), "True");
}

#[test]
fn eq_true() {
    let ctx = Context::new();
    assert_eq!(
        format!("{}", ctx.int(7).eq_expr(&ctx.int(7)).eval()),
        "True"
    );
}

#[test]
fn eq_false() {
    let ctx = Context::new();
    assert_eq!(
        format!("{}", ctx.int(7).eq_expr(&ctx.int(8)).eval()),
        "False"
    );
}

#[test]
fn ne_true() {
    let ctx = Context::new();
    assert_eq!(
        format!("{}", ctx.int(1).ne_expr(&ctx.int(2)).eval()),
        "True"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Logical connectives
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn and_true_true() {
    let ctx = Context::new();
    let t1 = ctx.int(5).gt(&ctx.int(3)); // True
    let t2 = ctx.int(7).gt(&ctx.int(1)); // True
    assert_eq!(format!("{}", t1.and(&t2).eval()), "True");
}

#[test]
fn and_true_false() {
    let ctx = Context::new();
    let t = ctx.int(5).gt(&ctx.int(3)); // True
    let f = ctx.int(1).gt(&ctx.int(9)); // False
    assert_eq!(format!("{}", t.and(&f).eval()), "False");
}

#[test]
fn or_false_true() {
    let ctx = Context::new();
    let f = ctx.int(1).gt(&ctx.int(9)); // False
    let t = ctx.int(5).gt(&ctx.int(3)); // True
    assert_eq!(format!("{}", f.or(&t).eval()), "True");
}

#[test]
fn or_false_false() {
    let ctx = Context::new();
    let f1 = ctx.int(1).gt(&ctx.int(9));
    let f2 = ctx.int(2).gt(&ctx.int(8));
    assert_eq!(format!("{}", f1.or(&f2).eval()), "False");
}

#[test]
fn not_true() {
    let ctx = Context::new();
    let t = ctx.int(5).gt(&ctx.int(3));
    assert_eq!(format!("{}", t.not().eval()), "False");
}

#[test]
fn not_false() {
    let ctx = Context::new();
    let f = ctx.int(1).gt(&ctx.int(9));
    assert_eq!(format!("{}", f.not().eval()), "True");
}

#[test]
fn double_not() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.gt(&ctx.int(0));
    let double = cond.not().not();
    // Not(Not(x>0)) should simplify to x>0 via canon
    let s = format!("{double}");
    assert!(s.contains("x") && s.contains(">"), "double not: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic relationals (unevaluated)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn symbolic_gt_stays() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.gt(&ctx.int(0));
    let s = format!("{}", cond.eval());
    // x > 0 can't be evaluated without knowing x
    assert!(s.contains("x") && s.contains(">"), "symbolic: {s}");
}

#[test]
fn symbolic_and() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let cond = x.gt(&ctx.int(0)).and(&y.gt(&ctx.int(0)));
    let s = format!("{cond}");
    assert!(s.contains("x") && s.contains("y"), "and: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Piecewise
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn piecewise_basic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.gt(&ctx.int(0));
    let neg_x = -&x;
    let pw = Ex::piecewise(&[(&x, &cond), (&neg_x, &cond.not())]);
    let s = format!("{pw}");
    assert!(s.contains("Piecewise"), "piecewise display: {s}");
}

#[test]
fn piecewise_diff() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.gt(&ctx.int(0));
    let pw = Ex::piecewise(&[(&x.powi(2), &cond), (&(-&x), &cond.not())]);
    let dpw = pw.diff(&x);
    let s = format!("{dpw}");
    assert!(s.contains("Piecewise"), "diff of piecewise: {s}");
}

#[test]
fn piecewise_eval_known_condition() {
    let ctx = Context::new();
    // Piecewise with a True condition should collapse
    let x = ctx.symbol("x");
    let t = ctx.int(5).gt(&ctx.int(3)); // True
    let pw = Ex::piecewise(&[(&x, &t)]);
    let evald = pw.eval();
    assert_eq!(format!("{evald}"), "x");
}

// ═══════════════════════════════════════════════════════════════════════════
// expr! macro with comparisons and logic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_gt() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond: BoolEx = expr!(ctx, x > 0);
    let s = format!("{cond}");
    assert!(s.contains(">"), "expr!(ctx, x > 0): {s}");
}

#[test]
fn expr_macro_lt() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond: BoolEx = expr!(ctx, x < 5);
    let s = format!("{cond}");
    assert!(s.contains("x") && s.contains("5"), "expr!(ctx, x < 5): {s}");
}

#[test]
fn expr_macro_le() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond: BoolEx = expr!(ctx, x <= 3);
    let s = format!("{cond}");
    assert!(s.contains("x"), "expr!(ctx, x <= 3): {s}");
}

#[test]
fn expr_macro_and() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let cond: BoolEx = expr!(ctx, x > 0 && y > 0);
    let s = format!("{cond}");
    assert!(s.contains("x") && s.contains("y"), "and: {s}");
}

#[test]
fn expr_macro_or() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let cond: BoolEx = expr!(ctx, x > 0 || y > 0);
    let s = format!("{cond}");
    assert!(s.contains("x") && s.contains("y"), "or: {s}");
}

#[test]
fn expr_macro_not() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond: BoolEx = expr!(ctx, !(x > 0));
    let s = format!("{cond}");
    assert!(s.contains("x"), "not: {s}");
}

#[test]
fn expr_macro_gt_eval() {
    let ctx = Context::new();
    let x = ctx.int(5);
    let result = expr!(ctx, x > 3).eval();
    assert_eq!(format!("{result}"), "True");
}

#[test]
fn expr_macro_and_eval() {
    let ctx = Context::new();
    let x = ctx.int(5);
    let y = ctx.int(3);
    let result = expr!(ctx, x > 0 && y > 0).eval();
    assert_eq!(format!("{result}"), "True");
}

#[test]
fn expr_macro_or_eval() {
    let ctx = Context::new();
    let x = ctx.int(5);
    let y = ctx.int(-3);
    let result = expr!(ctx, x > 0 || y > 0).eval();
    assert_eq!(format!("{result}"), "True");
}

#[test]
fn expr_macro_not_eval() {
    let ctx = Context::new();
    let x = ctx.int(5);
    let result = expr!(ctx, !(x > 10)).eval();
    assert_eq!(format!("{result}"), "True");
}

// ═══════════════════════════════════════════════════════════════════════════
// Type safety — these patterns should NOT compile (documented, not tested)
// ═══════════════════════════════════════════════════════════════════════════

// The following would be COMPILE ERRORS with phantom types:
// let _ = cond + 1;          // BoolEx + i64 not implemented
// let _ = cond.sin();        // sin() not on BoolEx
// let _ = x.and(&cond);      // and() not on Ex

// ═══════════════════════════════════════════════════════════════════════════
// BoolEx escape hatches
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn boolx_into_ex() {
    let ctx = Context::new();
    let cond = ctx.int(5).gt(&ctx.int(3));
    let ex: Ex = cond.into_ex();
    let _ = format!("{ex}"); // should display fine
}

#[test]
fn boolx_as_ex() {
    let ctx = Context::new();
    let cond = ctx.int(5).gt(&ctx.int(3));
    let ex: Ex = cond.as_ex();
    let _ = format!("{ex}");
}

#[test]
fn boolx_subs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.gt(&ctx.int(0));
    let substituted = cond.subs(&x, &ctx.int(5));
    assert_eq!(format!("{}", substituted.eval()), "True");
}

#[test]
fn boolx_free_symbols() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let cond = x.gt(&y);
    let syms = cond.free_symbols();
    assert_eq!(syms.len(), 2);
}
