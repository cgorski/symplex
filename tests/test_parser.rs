//! Tests for the runtime expression parser via symplex::parse::parse().

use symplex::prelude::*;

fn p(input: &str) -> String {
    let ctx = Context::new();
    let ex = symplex::parse::parse(&ctx, input).unwrap();
    format!("{ex}")
}

fn p_err(input: &str) -> bool {
    let ctx = Context::new();
    symplex::parse::parse(&ctx, input).is_err()
}

// ═══════════════════════════════════════════════════════════════════════════
// Atoms
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parse_integer() {
    assert_eq!(p("42"), "42");
}

#[test]
fn parse_zero() {
    assert_eq!(p("0"), "0");
}

#[test]
fn parse_symbol() {
    assert_eq!(p("x"), "x");
}

#[test]
fn parse_pi() {
    assert_eq!(p("pi"), "pi");
}

#[test]
fn parse_e() {
    assert_eq!(p("E"), "E");
}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parse_add() {
    assert_eq!(p("x + y"), "x + y");
}

#[test]
fn parse_sub() {
    assert_eq!(p("x - y"), "x - y");
}

#[test]
fn parse_mul() {
    assert_eq!(p("x * y"), "x*y");
}

#[test]
fn parse_div() {
    let s = p("x / y");
    assert!(s.contains("/y"), "should show division: {s}");
}

#[test]
fn parse_power() {
    assert_eq!(p("x^2"), "x^2");
}

#[test]
fn parse_neg() {
    assert_eq!(p("-x"), "-x");
}

// ═══════════════════════════════════════════════════════════════════════════
// Precedence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parse_precedence_add_mul() {
    // 2 + 3*x should be 2 + (3*x)
    let s = p("2 + 3*x");
    assert!(s.contains("3*x"), "mul should bind tighter than add: {s}");
}

#[test]
fn parse_precedence_parens() {
    let s = p("(x + 1)^2");
    assert!(
        s.contains("(1 + x)^2") || s.contains("(x + 1)^2"),
        "got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parse_sin() {
    assert_eq!(p("sin(x)"), "sin(x)");
}

#[test]
fn parse_cos() {
    assert_eq!(p("cos(x)"), "cos(x)");
}

#[test]
fn parse_exp() {
    assert_eq!(p("exp(x)"), "exp(x)");
}

#[test]
fn parse_ln() {
    assert_eq!(p("ln(x)"), "ln(x)");
}

#[test]
fn parse_sqrt() {
    assert_eq!(p("sqrt(x)"), "sqrt(x)");
}

#[test]
fn parse_nested_function() {
    assert_eq!(p("sin(cos(x))"), "sin(cos(x))");
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parse_polynomial() {
    let s = p("x^2 + 2*x + 1");
    assert!(s.contains("x^2") && s.contains("2*x"), "got: {s}");
}

#[test]
fn parse_trig_identity() {
    let ctx = Context::new();
    let ex = symplex::parse::parse(&ctx, "sin(x)^2 + cos(x)^2").unwrap();
    let simplified = ex.simplify();
    assert_eq!(format!("{simplified}"), "1");
}

#[test]
fn parse_and_differentiate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let ex = symplex::parse::parse(&ctx, "x^3").unwrap();
    let deriv = ex.diff(&x);
    assert_eq!(format!("{deriv}"), "3*x^2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Errors
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parse_empty_is_error() {
    assert!(p_err(""));
}

#[test]
fn parse_unknown_function_is_error() {
    assert!(p_err("foo(x)"));
}

#[test]
fn parse_unbalanced_parens_is_error() {
    assert!(p_err("(x + 1"));
}

#[test]
fn parse_trailing_operator_is_error() {
    assert!(p_err("x +"));
}
