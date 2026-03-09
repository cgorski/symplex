//! Integration tests for Apply chain-rule differentiation (Wave C)
//! and dependency-aware differentiation (Wave D).
//!
//! Wave C tests exercise the Apply arm of `diff_node` through the
//! public `Ex::diff()` / `Ex::fibonacci()` / `Ex::lucas()` API.
//!
//! Wave D unit tests for `diff_with_deps` and `eval_derivatives` live
//! inside their respective source modules (`diff.rs`, `subs.rs`)
//! because those modules are `pub(crate)`.

mod common;

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn check(expr: &Ex, expected: &str) {
    let s = format!("{expr}");
    assert_eq!(s, expected, "expected '{expected}', got '{s}'");
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave C — Apply chain rule (known Apply functions via public API)
// ═══════════════════════════════════════════════════════════════════════════

/// d/dx(fibonacci(10)) = 0 — known Apply function with constant arg.
#[test]
fn diff_known_apply_constant_fibonacci() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let n = __ctx.int(10);
    let fib = n.fibonacci();
    let result = fib.diff(&x);
    check(&result, "0");
}

/// d/dx(lucas(5)) = 0 — known Apply function with constant arg.
#[test]
fn diff_known_apply_constant_lucas() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let n = __ctx.int(5);
    let luc = n.lucas();
    let result = luc.diff(&x);
    check(&result, "0");
}

/// d/dx(fibonacci(x)) stays as a formal derivative since we can't
/// evaluate the derivative of fibonacci at arbitrary x.
#[test]
fn diff_known_apply_variable_fibonacci() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let fib = x.fibonacci();
    let result = fib.diff(&x);
    let s = format!("{result}");
    assert!(
        s.contains("Derivative") && s.contains("fibonacci"),
        "d/dx(fibonacci(x)) should be a formal Derivative, got '{s}'"
    );
}

/// d/dx(lucas(x)) stays as a formal derivative.
#[test]
fn diff_known_apply_variable_lucas() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let luc = x.lucas();
    let result = luc.diff(&x);
    let s = format!("{result}");
    assert!(
        s.contains("Derivative") && s.contains("lucas"),
        "d/dx(lucas(x)) should be a formal Derivative, got '{s}'"
    );
}

/// d/dx(fibonacci(y)) = 0 — y is independent of x, so chain rule
/// gives zero even though the argument is a symbol.
#[test]
fn diff_known_apply_other_symbol() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let y = __ctx.symbol("y");
    let fib = y.fibonacci();
    let result = fib.diff(&x);
    check(&result, "0");
}

/// d/dx(fibonacci(x²)) should apply the chain rule:
/// Derivative(fibonacci(x²), x²) · 2x
/// The result should contain "2", "x", and "Derivative".
#[test]
fn diff_known_apply_chain_rule_x_squared() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let x_sq = x.powi(2);
    let fib = x_sq.fibonacci();
    let result = fib.diff(&x);
    let s = format!("{result}");
    assert!(
        s.contains("2") && s.contains("x") && s.contains("Derivative"),
        "d/dx(fibonacci(x²)) should contain 2, x, and Derivative, got '{s}'"
    );
}

/// d/dx(fibonacci(sin(x))) should include cos(x) factor from chain rule.
#[test]
fn diff_known_apply_chain_sin() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let sin_x = x.sin();
    let fib = sin_x.fibonacci();
    let result = fib.diff(&x);
    let s = format!("{result}");
    assert!(
        s.contains("cos"),
        "d/dx(fibonacci(sin(x))) should contain cos(x) factor, got '{s}'"
    );
}

/// d/dx(fibonacci(exp(x))) should include exp(x) factor from chain rule.
#[test]
fn diff_known_apply_chain_exp() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let exp_x = x.exp();
    let fib = exp_x.fibonacci();
    let result = fib.diff(&x);
    let s = format!("{result}");
    assert!(
        s.contains("exp"),
        "d/dx(fibonacci(exp(x))) should contain exp(x) factor, got '{s}'"
    );
    assert!(
        s.contains("Derivative"),
        "d/dx(fibonacci(exp(x))) should contain formal Derivative, got '{s}'"
    );
}

/// d/dx(fibonacci(3*x + 1)) should have chain rule factor 3.
#[test]
fn diff_known_apply_chain_linear() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let three = __ctx.int(3);
    let one = __ctx.int(1);
    let arg = &three * &x + &one;
    let fib = arg.fibonacci();
    let result = fib.diff(&x);
    let s = format!("{result}");
    assert!(
        s.contains("3") && s.contains("Derivative"),
        "d/dx(fibonacci(3x+1)) should contain 3 and Derivative, got '{s}'"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression tests — existing diff behaviour is preserved
// ═══════════════════════════════════════════════════════════════════════════

/// Verify basic power rule still works after Apply chain rule changes.
#[test]
fn regression_power_rule() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    check(&x.powi(3).diff(&x), "3*x^2");
}

/// Second derivative still works correctly.
#[test]
fn regression_second_derivative() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let expr = x.powi(4);
    let first = expr.diff(&x);
    check(&first, "4*x^3");
    let second = first.diff(&x);
    check(&second, "12*x^2");
}

/// Chain rule for elementary functions is unchanged.
#[test]
fn regression_elementary_chain_rule() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    // d/dx(sin(x²)) = 2x·cos(x²)
    let expr = x.powi(2).sin();
    let result = expr.diff(&x);
    let s = format!("{result}");
    assert!(
        s.contains("cos") && s.contains("2"),
        "d/dx(sin(x²)) should contain cos and 2, got '{s}'"
    );
}

/// d/dx(exp(x)) = exp(x).
#[test]
fn regression_exp_diff() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    check(&x.exp().diff(&x), "exp(x)");
}

/// d/dx(ln(x)) = 1/x.
#[test]
fn regression_ln_diff() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    check(&x.ln().diff(&x), "1/x");
}

/// d/dx(sin(x)) = cos(x).
#[test]
fn regression_sin_diff() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    check(&x.sin().diff(&x), "cos(x)");
}

/// d/dx(cos(x)) = -sin(x).
#[test]
fn regression_cos_diff() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    check(&x.cos().diff(&x), "-sin(x)");
}

/// d/dx(x * sin(x)) = sin(x) + x*cos(x).
#[test]
fn regression_product_rule() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let expr = &x * &x.sin();
    let result = expr.diff(&x);
    let s = format!("{result}");
    assert!(
        s.contains("sin") && s.contains("cos"),
        "d/dx(x*sin(x)) should contain sin and cos, got '{s}'"
    );
}

/// d/dx(constant) = 0.
#[test]
fn regression_constant_diff() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let five = __ctx.int(5);
    check(&five.diff(&x), "0");
}
