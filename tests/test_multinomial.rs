//! Tests for multinomial theorem expansion.
//!
//! Validates that `expand()` correctly uses the binomial / multinomial
//! theorem instead of repeated multiplication for `(Add)^n`.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Binomial expansion — basic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn binomial_expand_squared() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; a, b);
    let expr = (&a + &b).powi(2).expand();
    // (a+b)² = a² + 2ab + b²  →  3 terms
    assert_eq!(expr.term_count(), 3, "got: {expr}");
}

#[test]
fn binomial_expand_cubed() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; a, b);
    let expr = (&a + &b).powi(3).expand();
    // (a+b)³ = a³ + 3a²b + 3ab² + b³  →  4 terms
    assert_eq!(expr.term_count(), 4, "got: {expr}");
}

#[test]
fn binomial_expand_20() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; a, b);
    let expr = (&a + &b).powi(20).expand();
    // (a+b)^20 has 21 terms
    assert_eq!(
        expr.term_count(),
        21,
        "(a+b)^20 should have 21 terms, got: {expr}"
    );
}

#[test]
fn binomial_x_plus_1_squared_display() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; x);
    let expr = (&x + 1).powi(2).expand();
    assert_eq!(format!("{expr}"), "x^2 + 2*x + 1");
}

#[test]
fn binomial_x_plus_1_cubed_display() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; x);
    let expr = (&x + 1).powi(3).expand();
    assert_eq!(format!("{expr}"), "x^3 + 3*x^2 + 3*x + 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Trinomial expansion
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn trinomial_expand_squared() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; a, b, c);
    let expr = (&a + &b + &c).powi(2).expand();
    // (a+b+c)² = a² + b² + c² + 2ab + 2ac + 2bc  →  6 terms
    assert_eq!(
        expr.term_count(),
        6,
        "(a+b+c)^2 should have 6 terms, got: {expr}"
    );
}

#[test]
fn trinomial_expand_cubed() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; a, b, c);
    let expr = (&a + &b + &c).powi(3).expand();
    // (a+b+c)³ has C(3+2, 2) = 10 terms
    assert_eq!(
        expr.term_count(),
        10,
        "(a+b+c)^3 should have 10 terms, got: {expr}"
    );
}

#[test]
fn trinomial_expand_4() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; a, b, c);
    let expr = (&a + &b + &c).powi(4).expand();
    // (a+b+c)^4 has C(4+2, 2) = 15 terms
    assert_eq!(
        expr.term_count(),
        15,
        "(a+b+c)^4 should have 15 terms, got: {expr}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Quadrinomial expansion
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quadrinomial_expand_cubed() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; a, b, c, d);
    let expr = (&a + &b + &c + &d).powi(3).expand();
    // (a+b+c+d)^3 has C(3+3, 3) = 20 terms
    assert_eq!(
        expr.term_count(),
        20,
        "(a+b+c+d)^3 should have 20 terms, got: {expr}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Correctness via numeric substitution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn binomial_correctness_via_substitution() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let orig = (&a + &b).powi(7);
    let expanded = orig.expand();

    // Evaluate both at a=3, b=5: (3+5)^7 = 8^7 = 2097152
    let three = ctx.int(3);
    let five = ctx.int(5);
    let orig_val = orig.subs(&a, &three).subs(&b, &five).eval();
    let exp_val = expanded.subs(&a, &three).subs(&b, &five).eval();
    assert_eq!(
        format!("{orig_val}"),
        format!("{exp_val}"),
        "expanded form should evaluate identically"
    );
    assert_eq!(format!("{orig_val}"), "2097152");
}

#[test]
fn trinomial_correctness_via_substitution() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");
    let orig = (&a + &b + &c).powi(5);
    let expanded = orig.expand();

    // Evaluate at a=2, b=3, c=4: (2+3+4)^5 = 9^5 = 59049
    let two = ctx.int(2);
    let three = ctx.int(3);
    let four = ctx.int(4);
    let orig_val = orig.subs(&a, &two).subs(&b, &three).subs(&c, &four).eval();
    let exp_val = expanded
        .subs(&a, &two)
        .subs(&b, &three)
        .subs(&c, &four)
        .eval();
    assert_eq!(
        format!("{orig_val}"),
        format!("{exp_val}"),
        "trinomial expanded form should evaluate identically"
    );
    assert_eq!(format!("{orig_val}"), "59049");
}

#[test]
fn quadrinomial_correctness_via_substitution() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");
    let d = ctx.symbol("d");
    let orig = (&a + &b + &c + &d).powi(4);
    let expanded = orig.expand();

    // Evaluate at a=1, b=2, c=3, d=4: (1+2+3+4)^4 = 10^4 = 10000
    let one = ctx.int(1);
    let two = ctx.int(2);
    let three = ctx.int(3);
    let four = ctx.int(4);
    let orig_val = orig
        .subs(&a, &one)
        .subs(&b, &two)
        .subs(&c, &three)
        .subs(&d, &four)
        .eval();
    let exp_val = expanded
        .subs(&a, &one)
        .subs(&b, &two)
        .subs(&c, &three)
        .subs(&d, &four)
        .eval();
    assert_eq!(
        format!("{orig_val}"),
        format!("{exp_val}"),
        "quadrinomial expanded form should evaluate identically"
    );
    assert_eq!(format!("{orig_val}"), "10000");
}

// ═══════════════════════════════════════════════════════════════════════════
// Idempotence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn multinomial_expand_is_idempotent() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; a, b, c);
    let first = (&a + &b + &c).powi(4).expand();
    let second = first.expand();
    assert_eq!(
        format!("{first}"),
        format!("{second}"),
        "expand should be idempotent for multinomial"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Performance: large binomial should be fast with multinomial theorem
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_binomial_50_is_fast() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; a, b);
    let start = std::time::Instant::now();
    let expr = (&a + &b).powi(50).expand();
    let elapsed = start.elapsed();
    assert_eq!(expr.term_count(), 51, "(a+b)^50 should have 51 terms");
    assert!(
        elapsed.as_millis() < 2000,
        "expand (a+b)^50 should complete quickly, took {:?}",
        elapsed
    );
}

#[test]
fn expand_binomial_100_term_count() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; a, b);
    let expr = (&a + &b).powi(100).expand();
    assert_eq!(expr.term_count(), 101, "(a+b)^100 should have 101 terms");
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic-coefficient binomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn binomial_with_numeric_coefficients() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");

    // (2a + 3b)^3 expanded, then evaluate at a=1, b=1 → (2+3)^3 = 125
    let orig = (&a * 2 + &b * 3).powi(3);
    let expanded = orig.expand();

    let one = ctx.int(1);
    let orig_val = orig.subs(&a, &one).subs(&b, &one).eval();
    let exp_val = expanded.subs(&a, &one).subs(&b, &one).eval();
    assert_eq!(format!("{orig_val}"), "125");
    assert_eq!(format!("{orig_val}"), format!("{exp_val}"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_pow_one_is_identity() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; a, b, c);
    let sum = &a + &b + &c;
    let expr = sum.powi(1);
    let expanded = expr.expand();
    // (a + b + c)^1 should canonicalize / reduce to a + b + c
    assert_eq!(expanded.term_count(), 3, "got: {expanded}");
}

#[test]
fn expand_single_term_power() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; x);
    // (x)^5 — base is not an Add, should stay x^5
    let expr = x.powi(5).expand();
    assert_eq!(format!("{expr}"), "x^5");
}

#[test]
fn expand_negative_power_not_expanded() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; a, b);
    let expr = (&a + &b).powi(-3).expand();
    let s = format!("{expr}");
    // Should stay as (a + b)^(-3) — negative powers are not expanded
    assert!(
        s.contains("^(-3)") || s.contains("^-3"),
        "negative power should not be expanded, got: {s}"
    );
}

#[test]
fn expand_binomial_correctness_high_power() {
    // Verify (x + 1)^10 evaluated at x=1 gives 2^10 = 1024
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let orig = (&x + 1).powi(10);
    let expanded = orig.expand();
    assert_eq!(expanded.term_count(), 11, "(x+1)^10 should have 11 terms");

    let one = ctx.int(1);
    let val = expanded.subs(&x, &one).eval();
    assert_eq!(format!("{val}"), "1024");
}

#[test]
fn trinomial_x_y_1_squared() {
    let __vars_ctx = symplex::default_context().clone(); symplex::syms!(__vars_ctx; x, y);
    let expr = (&x + &y + 1).powi(2).expand();
    // (x + y + 1)^2 = x^2 + y^2 + 1 + 2xy + 2x + 2y  →  6 terms
    assert_eq!(
        expr.term_count(),
        6,
        "(x+y+1)^2 should have 6 terms, got: {expr}"
    );
}
