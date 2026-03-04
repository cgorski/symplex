//! Tests for new functions in the expr! macro (Waves A, O, R).

use symplex::prelude::*;
use symplex::vars;

// ═══════════════════════════════════════════════════════════════════════════
// Wave A: reciprocal trig / hyperbolic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_sec() {
    vars!(x);
    let result = expr!(sec(x));
    assert_eq!(result, x.sec());
}

#[test]
fn expr_macro_csc() {
    vars!(x);
    let result = expr!(csc(x));
    assert_eq!(result, x.csc());
}

#[test]
fn expr_macro_cot() {
    vars!(x);
    let result = expr!(cot(x));
    assert_eq!(result, x.cot());
}

#[test]
fn expr_macro_sinc() {
    vars!(x);
    let result = expr!(sinc(x));
    assert_eq!(result, x.sinc());
}

#[test]
fn expr_macro_cosh_coth_sech_csch() {
    vars!(x);
    assert_eq!(expr!(coth(x)), x.coth());
    assert_eq!(expr!(sech(x)), x.sech());
    assert_eq!(expr!(csch(x)), x.csch());
}

#[test]
fn expr_macro_inverse_reciprocal_trig() {
    vars!(x);
    assert_eq!(expr!(acot(x)), x.acot());
    assert_eq!(expr!(asec(x)), x.asec());
    assert_eq!(expr!(acsc(x)), x.acsc());
}

#[test]
fn expr_macro_inverse_reciprocal_hyp() {
    vars!(x);
    assert_eq!(expr!(acoth(x)), x.acoth());
    assert_eq!(expr!(asech(x)), x.asech());
    assert_eq!(expr!(acsch(x)), x.acsch());
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave O: complex
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_conjugate() {
    vars!(x);
    let result = expr!(conjugate(x));
    assert_eq!(result, x.conjugate());
}

#[test]
fn expr_macro_arg() {
    vars!(x);
    let result = expr!(arg(x));
    assert_eq!(result, x.arg());
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave A: atan2 (multi-arg)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_atan2() {
    vars!(y, x);
    let result = expr!(atan2(y, x));
    assert_eq!(result, y.atan2(&x));
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave R: combinatorial (1-arg)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_fibonacci() {
    let n = symplex::int(10);
    let result = expr!(fibonacci(n));
    let evaled = result.eval();
    assert_eq!(format!("{evaled}"), "55");
}

#[test]
fn expr_macro_catalan() {
    let n = symplex::int(4);
    let result = expr!(catalan_number(n));
    assert_eq!(format!("{}", result.eval()), "14");
}

#[test]
fn expr_macro_bernoulli() {
    let n = symplex::int(2);
    let result = expr!(bernoulli_number(n));
    assert_eq!(format!("{}", result.eval()), "1/6");
}

#[test]
fn expr_macro_rising_factorial() {
    vars!(x);
    let n = symplex::int(3);
    let result = expr!(rising_factorial(x, n));
    assert_eq!(result, x.rising_factorial(&n));
}

#[test]
fn expr_macro_falling_factorial() {
    vars!(x);
    let n = symplex::int(3);
    let result = expr!(falling_factorial(x, n));
    assert_eq!(result, x.falling_factorial(&n));
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex expressions mixing new functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_complex_expression() {
    vars!(x);
    // sec(x)^2 + csc(x)^2 — uses both new trig functions
    let result = expr!(sec(x) ^ 2 + csc(x) ^ 2);
    let expected = &x.sec().powi(2) + &x.csc().powi(2);
    assert_eq!(result, expected);
}
