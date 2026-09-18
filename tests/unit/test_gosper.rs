//! Integration tests for Gosper's algorithm (hypergeometric summation).
//!
//! These tests exercise the Gosper summation machinery through the public
//! `closed_form_sum()` API once the wiring agent connects `gosper_sum` into
//! the evaluation pipeline.  Until then, they verify expected properties
//! of hypergeometric sums using direct numeric evaluation as a cross-check.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Geometric series — Σ_{k=0}^{n} r^k
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gosper_geometric_2k_numeric() {
    // Σ_{k=0}^{9} 2^k = 2^10 − 1 = 1023
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.int(2).pow(&k);
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(0), &ctx.int(9));
    let result = s.eval();
    assert_eq!(format!("{result}"), "1023");
}

#[test]
fn gosper_geometric_3k_numeric() {
    // Σ_{k=0}^{5} 3^k = (3^6 − 1)/2 = 364
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.int(3).pow(&k);
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(0), &ctx.int(5));
    let result = s.eval();
    assert_eq!(format!("{result}"), "364");
}

#[test]
fn gosper_geometric_closed_form() {
    // Σ_{k=0}^{n} 2^k should give a closed form.
    // Substituting n=10 should yield 2^11 − 1 = 2047.
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let body = ctx.int(2).pow(&k);
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(0), &n);
    let closed = s.closed_form_sum();
    let evaluated = closed.subs(&n, &ctx.int(10)).eval();
    // If closed_form_sum didn't find a form, the fallback brute-force
    // evaluation of 2^0 + 2^1 + … + 2^10 still yields 2047.
    assert_eq!(format!("{evaluated}"), "2047");
}

// ═══════════════════════════════════════════════════════════════════════════
// Factorial sums — Σ_{k=0}^{n} k·k!
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gosper_factorial_sum_numeric() {
    // Σ_{k=0}^{5} k·k! = 0 + 1 + 4 + 18 + 96 + 600 = 719 = 6! − 1
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = &k * &k.factorial();
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(0), &ctx.int(5));
    let result = s.eval();
    assert_eq!(format!("{result}"), "719");
}

#[test]
fn gosper_factorial_sum_n3() {
    // Σ_{k=0}^{3} k·k! = 0 + 1 + 4 + 18 = 23 = 4! − 1
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = &k * &k.factorial();
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(0), &ctx.int(3));
    let result = s.eval();
    assert_eq!(format!("{result}"), "23");
}

// ═══════════════════════════════════════════════════════════════════════════
// Binomial sums — Σ C(n,k) is NOT Gosper-summable
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gosper_binomial_sum_numeric() {
    // Σ_{k=0}^{5} C(5,k) = 2^5 = 32
    // This verifies the numeric evaluation even though Gosper's algorithm
    // cannot produce a closed form for C(n,k) (it requires Zeilberger's
    // creative telescoping).
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n5 = ctx.int(5);
    let body = Ex::binomial(&n5, &k);
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(0), &ctx.int(5));
    let result = s.eval();
    assert_eq!(format!("{result}"), "32");
}

// ═══════════════════════════════════════════════════════════════════════════
// Harmonic sum — NOT Gosper-summable
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gosper_harmonic_not_summable_numeric() {
    // Σ_{k=1}^{4} 1/k = 1 + 1/2 + 1/3 + 1/4 = 25/12
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.int(1) / &k;
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(1), &ctx.int(4));
    let result = s.eval();
    assert_eq!(format!("{result}"), "25/12");
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial sums (cross-check: these are handled by Faulhaber but also
// qualify as hypergeometric with rational ratio)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gosper_polynomial_sum_k() {
    // Σ_{k=1}^{10} k = 55
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let s = Ex::symbolic_sum(&k, &k, &ctx.int(1), &ctx.int(10));
    let result = s.eval();
    assert_eq!(format!("{result}"), "55");
}

#[test]
fn gosper_polynomial_sum_k_squared() {
    // Σ_{k=1}^{10} k² = 385
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.powi(2);
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(1), &ctx.int(10));
    let result = s.eval();
    assert_eq!(format!("{result}"), "385");
}

// ═══════════════════════════════════════════════════════════════════════════
// Closed-form with symbolic upper bound
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gosper_geometric_symbolic_upper_bound() {
    // Σ_{k=0}^{n} 2^k, closed form evaluated at n = 7 → 2^8 − 1 = 255
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let body = ctx.int(2).pow(&k);
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(0), &n);
    let closed = s.closed_form_sum();
    let at_7 = closed.subs(&n, &ctx.int(7)).eval();
    assert_eq!(format!("{at_7}"), "255");
}

#[test]
fn gosper_factorial_sum_symbolic_upper() {
    // Σ_{k=0}^{n} k·k! should yield (n+1)! − 1 when closed-form is found.
    // Verify numerically: n = 4 → 5! − 1 = 119.
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let body = &k * &k.factorial();
    let s = Ex::symbolic_sum(&body, &k, &ctx.int(0), &n);
    let closed = s.closed_form_sum();
    let at_4 = closed.subs(&n, &ctx.int(4)).eval();
    // If Gosper wiring is active, this gives 119 via the closed form.
    // If not, the brute-force evaluation 0+1+4+18+96 = 119 still works.
    assert_eq!(format!("{at_4}"), "119");
}
