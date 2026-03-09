//! Tests for evalf gaps: Sum, Product, Piecewise, Binomial.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Sum
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_finite_sum() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; k);
    let s = Ex::symbolic_sum(&k, &k, &__ctx.int(1), &__ctx.int(10));
    let result = s.eval_f64().unwrap();
    assert!(
        (result - 55.0).abs() < 1e-10,
        "Sum k=1..10 of k should be 55, got {result}"
    );
}

#[test]
fn evalf_sum_of_squares() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; k);
    let body = k.powi(2);
    let s = Ex::symbolic_sum(&body, &k, &__ctx.int(1), &__ctx.int(5));
    let result = s.eval_f64().unwrap();
    // 1 + 4 + 9 + 16 + 25 = 55
    assert!(
        (result - 55.0).abs() < 1e-10,
        "Sum k=1..5 of k^2 should be 55, got {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Product
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_finite_product() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; k);
    let p = Ex::symbolic_product(&k, &k, &__ctx.int(1), &__ctx.int(5));
    let result = p.eval_f64().unwrap();
    // 1 * 2 * 3 * 4 * 5 = 120
    assert!(
        (result - 120.0).abs() < 1e-10,
        "Product k=1..5 of k should be 120, got {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Binomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_binomial_5_2() {
    let __ctx = Context::new();
    // C(5,2) = 10
    let result = __ctx.int(5)
        .binomial(&__ctx.int(2))
        .eval_f64()
        .unwrap();
    assert!(
        (result - 10.0).abs() < 1e-6,
        "C(5,2) should be 10, got {result}"
    );
}

#[test]
fn evalf_binomial_10_3() {
    let __ctx = Context::new();
    // C(10,3) = 120
    let result = __ctx.int(10)
        .binomial(&__ctx.int(3))
        .eval_f64()
        .unwrap();
    assert!(
        (result - 120.0).abs() < 1e-6,
        "C(10,3) should be 120, got {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Piecewise
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_piecewise_true_branch() {
    let __ctx = Context::new();
    // Piecewise((42, True)) → 42
    let val = __ctx.int(42);
    let cond = __ctx.int(1).gt(&__ctx.int(0)); // 1 > 0 → True after eval
    let pw = Ex::piecewise(&[(&val, &cond)]);
    // eval() collapses 1>0 to BoolTrue, then piecewise selects the branch
    let result = pw.eval().eval_f64().unwrap();
    assert!(
        (result - 42.0).abs() < 1e-10,
        "Piecewise with True cond should be 42, got {result}"
    );
}

#[test]
fn piecewise_with_else_branch() {
    let __ctx = Context::new();
    // Piecewise with explicit True condition on last branch should work
    let val = __ctx.int(42);
    let cond = __ctx.int(1).gt(&__ctx.int(0)); // 1 > 0 → True
    let pw = Ex::piecewise(&[(&val, &cond)]);
    let result = pw.eval_f64();
    assert!(result.is_ok(), "piecewise with True condition should evaluate");
    assert!((result.unwrap() - 42.0).abs() < 1e-10);
}

#[test]
fn piecewise_all_false_returns_error() {
    let __ctx = Context::new();
    // Piecewise where all conditions are False should return Err
    let cond_f1 = __ctx.int(0).gt(&__ctx.int(1)); // 0 > 1 → False
    let cond_f2 = __ctx.int(0).gt(&__ctx.int(1)); // 0 > 1 → False
    let pw = Ex::piecewise(&[
        (&__ctx.int(1), &cond_f1),
        (&__ctx.int(2), &cond_f2),
    ]);
    let result = pw.eval_f64();
    assert!(result.is_err(), "piecewise with all-False conditions should return Err, got: {:?}", result);
}

#[test]
fn piecewise_first_true_wins() {
    let __ctx = Context::new();
    // First True condition should be selected
    let cond_t1 = __ctx.int(1).gt(&__ctx.int(0)); // 1 > 0 → True
    let cond_t2 = __ctx.int(1).gt(&__ctx.int(0)); // 1 > 0 → True
    let pw = Ex::piecewise(&[
        (&__ctx.int(1), &cond_t1),
        (&__ctx.int(2), &cond_t2),
    ]);
    let result = pw.eval_f64();
    assert!(result.is_ok());
    assert!((result.unwrap() - 1.0).abs() < 1e-10, "first True branch should win");
}
