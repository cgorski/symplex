//! Tests for evalf gaps: Sum, Product, Piecewise, Binomial.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Sum
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_finite_sum() {
    symplex::vars!(k);
    let s = Ex::symbolic_sum(&k, &k, &symplex::int(1), &symplex::int(10));
    let result = s.evalf_f64().unwrap();
    assert!(
        (result - 55.0).abs() < 1e-10,
        "Sum k=1..10 of k should be 55, got {result}"
    );
}

#[test]
fn evalf_sum_of_squares() {
    symplex::vars!(k);
    let body = k.powi(2);
    let s = Ex::symbolic_sum(&body, &k, &symplex::int(1), &symplex::int(5));
    let result = s.evalf_f64().unwrap();
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
    symplex::vars!(k);
    let p = Ex::symbolic_product(&k, &k, &symplex::int(1), &symplex::int(5));
    let result = p.evalf_f64().unwrap();
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
    // C(5,2) = 10
    let result = symplex::int(5)
        .binomial(&symplex::int(2))
        .evalf_f64()
        .unwrap();
    assert!(
        (result - 10.0).abs() < 1e-6,
        "C(5,2) should be 10, got {result}"
    );
}

#[test]
fn evalf_binomial_10_3() {
    // C(10,3) = 120
    let result = symplex::int(10)
        .binomial(&symplex::int(3))
        .evalf_f64()
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
    // Piecewise((42, True)) → 42
    let val = symplex::int(42);
    let cond = symplex::int(1).gt(&symplex::int(0)); // 1 > 0 → True after eval
    let pw = Ex::piecewise(&[(&val, &cond)]);
    // eval() collapses 1>0 to BoolTrue, then piecewise selects the branch
    let result = pw.eval().evalf_f64().unwrap();
    assert!(
        (result - 42.0).abs() < 1e-10,
        "Piecewise with True cond should be 42, got {result}"
    );
}
