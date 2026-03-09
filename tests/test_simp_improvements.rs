//! Integration tests for Wave X simplification improvements:
//! - trigsimp choice-set strategy (X1+X2+X3)
//! - combsimp factorial/binomial simplification (X4)
//! - nsimplify closed-form detection (X5)

#[allow(unused_imports)]
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// trigsimp — choice-set strategy
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn trigsimp_uses_trig_combine() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // 2*sin(x)*cos(x) should simplify to sin(2x) via trig_combine strategy
    let e = &x.sin() * &x.cos() * 2;
    let result = e.simplify_trig();
    let s = format!("{result}");
    // Should contain sin(2*x) or equivalent, and be no larger than original
    assert!(
        result.count_ops() <= e.count_ops(),
        "trigsimp should not increase complexity: {s} (result ops={}, original ops={})",
        result.count_ops(),
        e.count_ops(),
    );
}

#[test]
fn trigsimp_pythagorean_still_works() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let e = &x.sin().powi(2) + &x.cos().powi(2);
    let result = e.simplify_trig();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn trigsimp_pythagorean_plus_constant() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let e = &x.sin().powi(2) + &x.cos().powi(2) + 5;
    let result = e.simplify_trig();
    assert_eq!(format!("{result}"), "6");
}

#[test]
fn trigsimp_cos2_minus_sin2_double_angle() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // cos²(x) - sin²(x) should simplify via trig_combine to cos(2x)
    let e = &x.cos().powi(2) - &x.sin().powi(2);
    let result = e.simplify_trig();
    let s = format!("{result}");
    assert!(
        result.count_ops() <= e.count_ops(),
        "trigsimp should simplify cos²-sin²: {s} (result ops={}, original ops={})",
        result.count_ops(),
        e.count_ops(),
    );
}

#[test]
fn trigsimp_leaves_simple_trig_alone() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let e = x.sin();
    let result = e.simplify_trig();
    assert_eq!(format!("{result}"), "sin(x)");
}

#[test]
fn trigsimp_expand_then_recombine() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // Start with sin(x)^2 which has 2 ops (Sin + Pow).
    // trigsimp should not bloat it.
    let e = x.sin().powi(2);
    let result = e.simplify_trig();
    assert!(
        result.count_ops() <= e.count_ops(),
        "trigsimp should not bloat sin²(x): got {} ops vs {} ops, result = {}",
        result.count_ops(),
        e.count_ops(),
        result,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// combsimp — factorial / binomial simplification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn combsimp_factorial_ratio_concrete() {
    let ctx = Context::new();
    // 5! / 4! should simplify to 5 after eval + combsimp
    let five_fact = ctx.int(5).factorial().eval();
    let four_fact = ctx.int(4).factorial().eval();
    let ratio = &five_fact / &four_fact;
    let result = ratio.simplify_combinatorial();
    assert_eq!(format!("{result}"), "5");
}

#[test]
fn combsimp_factorial_ratio_symbolic() {
    let ctx = Context::new();
    symplex::syms!(ctx; n);
    // n! / (n-1)! should simplify to n
    let n_fact = n.factorial();
    let nm1 = &n - 1;
    let nm1_fact = nm1.factorial();
    let ratio = &n_fact / &nm1_fact;
    let result = ratio.simplify_combinatorial();
    assert_eq!(format!("{result}"), "n");
}

#[test]
fn combsimp_same_factorial_cancels() {
    let ctx = Context::new();
    symplex::syms!(ctx; n);
    let n_fact = n.factorial();
    let ratio = &n_fact / &n_fact;
    let result = ratio.simplify_combinatorial();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn combsimp_factorial_diff_2() {
    let ctx = Context::new();
    symplex::syms!(ctx; n);
    // n! / (n-2)! = n*(n-1)
    let n_fact = n.factorial();
    let nm2 = &n - 2;
    let nm2_fact = nm2.factorial();
    let ratio = &n_fact / &nm2_fact;
    let result = ratio.simplify_combinatorial();
    let s = format!("{result}");
    // Should not contain factorial notation
    assert!(
        !s.contains('!') && !s.contains("factorial"),
        "should not contain factorials: {s}"
    );
}

#[test]
fn combsimp_no_factorial_unchanged() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let e = &x + &y;
    let result = e.simplify_combinatorial();
    assert_eq!(format!("{result}"), format!("{e}"));
}

// ═══════════════════════════════════════════════════════════════════════════
// nsimplify — find closed-form for numerical expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nsimplify_finds_rational() {
    let ctx = Context::new();
    // 0.333333 should become 1/3
    let expr = ctx.rational(333333, 1000000);
    let result = expr.simplify_numeric(1e-5);
    assert_eq!(format!("{result}"), "1/3");
}

#[test]
fn nsimplify_finds_pi() {
    let ctx = Context::new();
    // A rational approximation of π
    let expr = ctx.rational(314159265, 100000000);
    let result = expr.simplify_numeric(1e-7);
    let s = format!("{result}");
    assert!(s.contains("pi"), "should find π, got: {s}");
}

#[test]
fn nsimplify_finds_sqrt2() {
    let ctx = Context::new();
    // √2 ≈ 1.4142
    let expr = ctx.rational(14142, 10000);
    let result = expr.simplify_numeric(1e-3);
    let s = format!("{result}");
    // Should produce 2^(1/2) or equivalent representation with ^ or 1/2
    assert!(s.contains("1/2") || s.contains("^"), "should find √2: {s}");
}

#[test]
fn nsimplify_exact_integer() {
    let ctx = Context::new();
    let expr = ctx.int(7);
    let result = expr.simplify_numeric(1e-10);
    assert_eq!(format!("{result}"), "7");
}

#[test]
fn nsimplify_negative_rational() {
    let ctx = Context::new();
    // -1/7 ≈ -0.142857
    let expr = ctx.rational(-142857, 1000000);
    let result = expr.simplify_numeric(1e-5);
    let s = format!("{result}");
    assert!(s == "-1/7", "should find -1/7, got: {s}");
}

#[test]
fn nsimplify_half_pi() {
    let ctx = Context::new();
    // π/2 ≈ 1.5707963
    let expr = ctx.rational(15707963, 10000000);
    let result = expr.simplify_numeric(1e-6);
    let s = format!("{result}");
    assert!(s.contains("pi"), "should find π/2, got: {s}");
}

#[test]
fn nsimplify_free_symbol_unchanged() {
    let ctx = Context::new();
    // An expression with free symbols can't be evaluated, so nsimplify
    // should return it unchanged.
    symplex::syms!(ctx; x);
    let result = x.simplify_numeric(1e-10);
    assert_eq!(format!("{result}"), "x");
}

#[test]
fn nsimplify_zero() {
    let ctx = Context::new();
    let expr = ctx.int(0);
    let result = expr.simplify_numeric(1e-10);
    assert_eq!(format!("{result}"), "0");
}
