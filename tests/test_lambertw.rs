//! Integration tests for the LambertW function and LambertW-based equation solving.
//!
//! The LambertW function W(x) satisfies W(x)·exp(W(x)) = x.
//!
//! # Note on solve tests
//!
//! The public `Ex::solve()` / `Ex::solve_or_empty()` API currently gates on
//! polynomial convertibility before dispatching to the internal solver.
//! Transcendental solvers (inversion peeling, change-of-variable, LambertW)
//! are exercised through the internal `solve::solve` function, which is
//! tested via unit tests in `src/solve.rs`.  The integration tests here
//! focus on LambertW *evaluation* and non-panic robustness through the
//! public API.

// ═══════════════════════════════════════════════════════════════════════════
// LambertW evaluation at known values
// ═══════════════════════════════════════════════════════════════════════════

use symplex::prelude::*;
#[test]
fn lambertw_eval_at_zero() {
    let __ctx = Context::new();
    // W(0) = 0 because 0·exp(0) = 0
    let result = __ctx.int(0).lambertw().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn lambertw_eval_at_e() {
    let __ctx = Context::new();
    // W(e) = 1 because 1·exp(1) = e
    let result = __ctx.e().lambertw().eval();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn lambertw_symbolic_stays_symbolic() {
    let __ctx = Context::new();
    // W(5) has no closed form — should remain as lambertw(5)
    let result = __ctx.int(5).lambertw().eval();
    let s = format!("{result}");
    assert!(
        s.contains("W("),
        "W(5) should stay symbolic, got: {s}"
    );
}

#[test]
fn lambertw_of_negative_stays_symbolic() {
    let __ctx = Context::new();
    // W(-1) has no simple closed form on the principal branch
    let result = __ctx.int(-1).lambertw().eval();
    let s = format!("{result}");
    assert!(
        s.contains("W("),
        "W(-1) should stay symbolic, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// LambertW symbolic construction and display
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lambertw_display_format() {
    let __ctx = Context::new();
    let expr = __ctx.int(3).lambertw();
    let s = format!("{expr}");
    assert!(
        s.contains("W(") && s.contains("3"),
        "display should show W(3), got: {s}"
    );
}

#[test]
fn lambertw_of_expression() {
    let __ctx = Context::new();
    // W(x + 1) should display sensibly and not panic
    let x = __ctx.symbol("x");
    let expr = (&x + 1).lambertw();
    let s = format!("{expr}");
    assert!(
        s.contains("W("),
        "W(x+1) should display with W, got: {s}"
    );
}

#[test]
fn lambertw_nested_eval() {
    let __ctx = Context::new();
    // W(W(e)) = W(1) — since W(e)=1, W(W(e)) = W(1) which stays symbolic
    let inner = __ctx.e().lambertw().eval(); // = 1
    let outer = inner.lambertw().eval(); // = W(1) ... but 1·exp(1) = e ≠ 1, so W(1) ≠ 1
    // W(1) ≈ 0.5671; stays symbolic since no closed form
    // But inner evaluated to 1, so this is W(1)
    let s = format!("{outer}");
    // W(1) should either stay as lambertw(1) or evaluate — either is fine
    assert!(
        s.contains("W(") || s.parse::<f64>().is_ok(),
        "W(W(e)) = W(1) should be representable, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Robustness: public solve_or_empty should not panic on transcendental eqs
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_or_empty_no_panic_on_x_exp_x() {
    let __ctx = Context::new();
    // x·exp(x) - 1 is not polynomial, so public solve_or_empty returns []
    // (the internal solver handles it — see unit tests in solve.rs).
    // Key assertion: it must not panic.
    let x = __ctx.symbol("x");
    let eq = &x * &x.exp() - 1;
    let _roots = eq.solve_or_empty(&x);
    // No panic = success
}

#[test]
fn solve_or_empty_no_panic_on_exp_plus_linear() {
    let __ctx = Context::new();
    // exp(x) + x - 2 is not polynomial
    let x = __ctx.symbol("x");
    let eq = x.exp() + &x - 2;
    let _roots = eq.solve_or_empty(&x);
    // No panic = success
}

#[test]
fn solve_or_empty_no_panic_on_x2_exp_x() {
    let __ctx = Context::new();
    // x²·exp(x) - 1 is not polynomial and not a LambertW pattern either
    let x = __ctx.symbol("x");
    let eq = x.powi(2) * &x.exp() - 1;
    let _roots = eq.solve_or_empty(&x);
    // No panic = success
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial solve still works (regression guard)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn polynomial_solve_unaffected() {
    let __ctx = Context::new();
    // x² - 1 = 0 → x = ±1 (polynomial path, must still work)
    let x = __ctx.symbol("x");
    let eq = x.powi(2) - 1;
    let roots = eq.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "x²-1 should still yield 2 roots");
    let mut vals: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    vals.sort();
    assert_eq!(vals, vec!["-1", "1"], "roots of x²-1: {vals:?}");
}
