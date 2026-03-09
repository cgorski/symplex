//! Tests for new functions in the expr! macro (Waves A, O, R).

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Wave A: reciprocal trig / hyperbolic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_sec() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = expr!(ctx, sec(x));
    assert_eq!(result, x.sec());
}

#[test]
fn expr_macro_csc() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = expr!(ctx, csc(x));
    assert_eq!(result, x.csc());
}

#[test]
fn expr_macro_cot() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = expr!(ctx, cot(x));
    assert_eq!(result, x.cot());
}

#[test]
fn expr_macro_sinc() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = expr!(ctx, sinc(x));
    assert_eq!(result, x.sinc());
}

#[test]
fn expr_macro_cosh_coth_sech_csch() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    assert_eq!(expr!(ctx, coth(x)), x.coth());
    assert_eq!(expr!(ctx, sech(x)), x.sech());
    assert_eq!(expr!(ctx, csch(x)), x.csch());
}

#[test]
fn expr_macro_inverse_reciprocal_trig() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    assert_eq!(expr!(ctx, acot(x)), x.acot());
    assert_eq!(expr!(ctx, asec(x)), x.asec());
    assert_eq!(expr!(ctx, acsc(x)), x.acsc());
}

#[test]
fn expr_macro_inverse_reciprocal_hyp() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    assert_eq!(expr!(ctx, acoth(x)), x.acoth());
    assert_eq!(expr!(ctx, asech(x)), x.asech());
    assert_eq!(expr!(ctx, acsch(x)), x.acsch());
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave O: complex
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_conjugate() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = expr!(ctx, conjugate(x));
    assert_eq!(result, x.conjugate());
}

#[test]
fn expr_macro_arg() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = expr!(ctx, arg(x));
    assert_eq!(result, x.arg());
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave A: atan2 (multi-arg)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_atan2() {
    let ctx = Context::new();
    symplex::syms!(ctx; y, x);
    let result = expr!(ctx, atan2(y, x));
    assert_eq!(result, y.atan2(&x));
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave R: combinatorial (1-arg)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_fibonacci() {
    let ctx = Context::new();
    let n = ctx.int(10);
    let result = expr!(ctx, fibonacci(n));
    let evaled = result.eval();
    assert_eq!(format!("{evaled}"), "55");
}

#[test]
fn expr_macro_catalan() {
    let ctx = Context::new();
    let n = ctx.int(4);
    let result = expr!(ctx, catalan_number(n));
    assert_eq!(format!("{}", result.eval()), "14");
}

#[test]
fn expr_macro_bernoulli() {
    let ctx = Context::new();
    let n = ctx.int(2);
    let result = expr!(ctx, bernoulli_number(n));
    assert_eq!(format!("{}", result.eval()), "1/6");
}

#[test]
fn expr_macro_rising_factorial() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let n = ctx.int(3);
    let result = expr!(ctx, rising_factorial(x, n));
    assert_eq!(result, x.rising_factorial(&n));
}

#[test]
fn expr_macro_falling_factorial() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let n = ctx.int(3);
    let result = expr!(ctx, falling_factorial(x, n));
    assert_eq!(result, x.falling_factorial(&n));
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex expressions mixing new functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_complex_expression() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // sec(x)^2 + csc(x)^2 — uses both new trig functions
    let result = expr!(ctx, sec(x) ^ 2 + csc(x) ^ 2);
    let expected = &x.sec().powi(2) + &x.csc().powi(2);
    assert_eq!(result, expected);
}
