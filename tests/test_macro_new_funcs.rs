//! Tests for new functions in the expr! macro (Waves A, O, R).

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Wave A: reciprocal trig / hyperbolic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_sec() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let result = expr!(sec(x));
    assert_eq!(result, x.sec());
}

#[test]
fn expr_macro_csc() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let result = expr!(csc(x));
    assert_eq!(result, x.csc());
}

#[test]
fn expr_macro_cot() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let result = expr!(cot(x));
    assert_eq!(result, x.cot());
}

#[test]
fn expr_macro_sinc() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let result = expr!(sinc(x));
    assert_eq!(result, x.sinc());
}

#[test]
fn expr_macro_cosh_coth_sech_csch() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    assert_eq!(expr!(coth(x)), x.coth());
    assert_eq!(expr!(sech(x)), x.sech());
    assert_eq!(expr!(csch(x)), x.csch());
}

#[test]
fn expr_macro_inverse_reciprocal_trig() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    assert_eq!(expr!(acot(x)), x.acot());
    assert_eq!(expr!(asec(x)), x.asec());
    assert_eq!(expr!(acsc(x)), x.acsc());
}

#[test]
fn expr_macro_inverse_reciprocal_hyp() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    assert_eq!(expr!(acoth(x)), x.acoth());
    assert_eq!(expr!(asech(x)), x.asech());
    assert_eq!(expr!(acsch(x)), x.acsch());
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave O: complex
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_conjugate() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let result = expr!(conjugate(x));
    assert_eq!(result, x.conjugate());
}

#[test]
fn expr_macro_arg() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let result = expr!(arg(x));
    assert_eq!(result, x.arg());
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave A: atan2 (multi-arg)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_atan2() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; y, x);
    let result = expr!(atan2(y, x));
    assert_eq!(result, y.atan2(&x));
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave R: combinatorial (1-arg)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_fibonacci() {
    let __ctx = Context::new();
    let n = __ctx.int(10);
    let result = expr!(fibonacci(n));
    let evaled = result.eval();
    assert_eq!(format!("{evaled}"), "55");
}

#[test]
fn expr_macro_catalan() {
    let __ctx = Context::new();
    let n = __ctx.int(4);
    let result = expr!(catalan_number(n));
    assert_eq!(format!("{}", result.eval()), "14");
}

#[test]
fn expr_macro_bernoulli() {
    let __ctx = Context::new();
    let n = __ctx.int(2);
    let result = expr!(bernoulli_number(n));
    assert_eq!(format!("{}", result.eval()), "1/6");
}

#[test]
fn expr_macro_rising_factorial() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let n = __ctx.int(3);
    let result = expr!(rising_factorial(x, n));
    assert_eq!(result, x.rising_factorial(&n));
}

#[test]
fn expr_macro_falling_factorial() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let n = __ctx.int(3);
    let result = expr!(falling_factorial(x, n));
    assert_eq!(result, x.falling_factorial(&n));
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex expressions mixing new functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_complex_expression() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    // sec(x)^2 + csc(x)^2 — uses both new trig functions
    let result = expr!(sec(x) ^ 2 + csc(x) ^ 2);
    let expected = &x.sec().powi(2) + &x.csc().powi(2);
    assert_eq!(result, expected);
}
