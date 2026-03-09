//! Integration tests for Wave E: closed-form sum evaluation, convergence tests,
//! and Gamma recurrence.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Faulhaber closed-form sums (numeric bounds)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn faulhaber_sum_k_1_to_10() {
    // Σ_{k=1}^{10} k = 55
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let s = Ex::symbolic_sum(&k, &k, &ctx.int(1), &ctx.int(10));
    let result = s.eval();
    assert_eq!(format!("{result}"), "55");
}

#[test]
fn faulhaber_sum_k_sq_1_to_10() {
    // Σ_{k=1}^{10} k² = 385
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.powi(2);
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(1), &ctx.int(10));
    let result = s.eval();
    assert_eq!(format!("{result}"), "385");
}

#[test]
fn faulhaber_sum_k_cube_1_to_10() {
    // Σ_{k=1}^{10} k³ = 3025
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.powi(3);
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(1), &ctx.int(10));
    let result = s.eval();
    assert_eq!(format!("{result}"), "3025");
}

#[test]
fn faulhaber_sum_k_4_1_to_10() {
    // Σ_{k=1}^{10} k⁴ = 25333
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.powi(4);
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(1), &ctx.int(10));
    let result = s.eval();
    assert_eq!(format!("{result}"), "25333");
}

// ═══════════════════════════════════════════════════════════════════════════
// Geometric series
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn geometric_sum_2_pow_k_0_to_9() {
    // Σ_{k=0}^{9} 2^k = 2^0 + 2^1 + ... + 2^9 = 1023
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.int(2).pow(&k);
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(0), &ctx.int(9));
    let result = s.eval();
    assert_eq!(format!("{result}"), "1023");
}

// ═══════════════════════════════════════════════════════════════════════════
// Constant body sum
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn constant_sum_5_from_1_to_10() {
    // Σ_{k=1}^{10} 5 = 50
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.int(5);
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(1), &ctx.int(10));
    let result = s.eval();
    assert_eq!(format!("{result}"), "50");
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic bounds (closed form with symbolic upper)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn symbolic_sum_k_1_to_n_gives_closed_form() {
    // Σ_{k=1}^{n} k should give a closed form: n*(n+1)/2
    // When we substitute n=10, it should give 55
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let s = Ex::symbolic_sum(&k, &k, &ctx.int(1), &n);
    let closed = s.closed_form_sum();
    // Verify by substituting n=10
    let evaluated = closed.subs(&n, &ctx.int(10)).eval();
    assert_eq!(format!("{evaluated}"), "55");
}

#[test]
fn symbolic_constant_sum_to_n() {
    // Σ_{k=1}^{n} 3 should give 3*n
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let body = ctx.int(3);
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(1), &n);
    let closed = s.closed_form_sum();
    // Verify by substituting n=10
    let evaluated = closed.subs(&n, &ctx.int(10)).eval();
    assert_eq!(format!("{evaluated}"), "30");
}

// ═══════════════════════════════════════════════════════════════════════════
// Convergence tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergence_p_series_k_sq_converges() {
    // Σ 1/k² converges (p-series with p=2 > 1)
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.powi(-2);
    assert_eq!(body.is_convergent(&k), Some(true));
}

#[test]
fn convergence_harmonic_diverges() {
    // Σ 1/k diverges (harmonic series, p=1)
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.powi(-1);
    assert_eq!(body.is_convergent(&k), Some(false));
}

#[test]
fn convergence_geometric_half_converges() {
    // Σ (1/2)^k converges (|r| < 1)
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let half = ctx.rational(1, 2);
    let body = half.pow(&k);
    assert_eq!(body.is_convergent(&k), Some(true));
}

#[test]
fn convergence_geometric_2_diverges() {
    // Σ 2^k diverges (|r| >= 1)
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.int(2).pow(&k);
    assert_eq!(body.is_convergent(&k), Some(false));
}

// ═══════════════════════════════════════════════════════════════════════════
// Gamma recurrence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gamma_positive_integer_still_works() {
    let __ctx = Context::new();
    // Gamma(5) = 4! = 24
    let result = __ctx.int(5).gamma().eval();
    assert_eq!(format!("{result}"), "24");
}

#[test]
fn gamma_half_integer_still_works() {
    let __ctx = Context::new();
    // Gamma(1/2) = √π
    let result = __ctx.rational(1, 2).gamma().eval();
    assert_eq!(format!("{result}"), "sqrt(pi)");
}

#[test]
fn gamma_recurrence_7_over_3() {
    let __ctx = Context::new();
    // Gamma(7/3) = (4/3)·(1/3)·Gamma(1/3)
    // = 4/9 · Gamma(1/3)
    let result = __ctx.rational(7, 3).gamma().eval();
    let display = format!("{result}");
    // Should contain Gamma(1/3) since that's the irreducible part
    assert!(
        display.contains("Gamma(1/3)") || display.contains("gamma(1/3)"),
        "Expected Gamma(1/3) in result, got: {display}"
    );
}

#[test]
fn gamma_recurrence_5_over_3() {
    let __ctx = Context::new();
    // Gamma(5/3) = (2/3)·Gamma(2/3)
    let result = __ctx.rational(5, 3).gamma().eval();
    let display = format!("{result}");
    // Should contain Gamma(2/3) since that's the irreducible part
    assert!(
        display.contains("Gamma(2/3)") || display.contains("gamma(2/3)"),
        "Expected Gamma(2/3) in result, got: {display}"
    );
}
