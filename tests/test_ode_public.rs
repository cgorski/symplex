//! Tests for ODE public API and new expr! functions.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// formal_diff — creates unevaluated Derivative node
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn formal_diff_creates_node() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let s = format!("{dy}");
    // Should display as some derivative notation, not evaluate to 0
    assert!(s != "0", "formal_diff should not evaluate: {s}");
}

#[test]
fn formal_diff_in_expression() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let expr = &dy + &y; // y' + y
    let s = format!("{expr}");
    assert!(s.contains("y"), "should contain y: {s}");
}

#[test]
fn formal_diff_via_expr_macro() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = expr!(diff(y, x));
    let s = format!("{dy}");
    assert!(s != "0", "expr!(diff(y,x)) should not evaluate: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// dsolve — ODE solving through public API
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dsolve_simple_separable() {
    // y' - x = 0 → y = x²/2 + C1
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let ode = &dy - &x; // y' - x = 0
    let sol = ode.try_solve_ode(&y, &x)
        .expect("dsolve should handle y' - x = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "should have constant: {s}");
    assert!(s.contains("x"), "should contain x: {s}");
}

#[test]
fn dsolve_exponential_decay() {
    // y' + 2y = 0 → y = C1*exp(-2x)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let dy = y.formal_diff(&x);
    let ode = &dy + &(&y * 2); // y' + 2y = 0
    let sol = ode.try_solve_ode(&y, &x)
        .expect("dsolve should handle y' + 2y = 0");
    let s = format!("{sol}");
    assert!(s.contains("exp"), "should contain exp: {s}");
    assert!(s.contains("C1"), "should have constant: {s}");
}

#[test]
fn dsolve_via_expr_macro() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let ode = expr!(diff(y, x) + 2 * y); // y' + 2y = 0
    let sol = ode.try_solve_ode(&y, &x)
        .expect("dsolve should handle y' + 2y = 0 via expr macro");
    let s = format!("{sol}");
    assert!(s.contains("exp") || s.contains("C1"), "solution: {s}");
}

#[test]
fn dsolve_dy_equals_zero() {
    // y' = 0 → y = C1
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let ode = y.formal_diff(&x); // y' = 0
    let sol = ode.try_solve_ode(&y, &x)
        .expect("dsolve should handle y' = 0");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "should be constant: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// factorial / binomial via Ex methods
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorial_via_method() {
    let result = symplex::default_context().int(5).factorial().eval();
    assert_eq!(format!("{result}"), "120");
}

#[test]
fn factorial_100_via_method() {
    let result = symplex::default_context().int(100).factorial().eval();
    let s = format!("{result}");
    assert!(
        s.starts_with("933262154"),
        "100! starts wrong: {}",
        &s[..20]
    );
    assert_eq!(s.len(), 158, "100! should have 158 digits");
}

#[test]
fn binomial_via_method() {
    let result = symplex::default_context().int(10).binomial(&symplex::default_context().int(3)).eval();
    assert_eq!(format!("{result}"), "120");
}

#[test]
fn factorial_via_expr_macro() {
    let result = expr!(factorial(5)).eval();
    assert_eq!(format!("{result}"), "120");
}

#[test]
fn binomial_via_expr_macro() {
    let result = expr!(C(10, 3)).eval();
    assert_eq!(format!("{result}"), "120");
}

#[test]
fn binomial_via_binomial_name() {
    let result = expr!(binomial(10, 5)).eval();
    assert_eq!(format!("{result}"), "252");
}

// ═══════════════════════════════════════════════════════════════════════════
// Context::neg_infinity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn neg_infinity_exists() {
    let ctx = symplex::default_context();
    let neg_inf = ctx.neg_infinity();
    assert_eq!(format!("{neg_inf}"), "-oo");
}

#[test]
fn limit_at_neg_infinity() {
    let x = symplex::default_context().symbol("x");
    let ctx = symplex::default_context();
    let result = (1 / &x).limit(&x, &ctx.neg_infinity());
    assert_eq!(format!("{result}"), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Combined workflows
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_via_eq_macro() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    // Build y' + y = 0 via eq! macro...
    // eq! doesn't support diff() yet, so build manually
    let ode = expr!(diff(y, x) + y);
    let sol = ode.try_solve_ode(&y, &x)
        .expect("dsolve should handle y' + y = 0");
    let s = format!("{sol}");
    assert!(s.contains("exp") || s.contains("C1"), "ODE solution: {s}");
}

#[test]
fn factorial_in_expression() {
    let n = symplex::default_context().symbol("n");
    let expr = n.factorial();
    let s = format!("{expr}");
    assert!(s.contains("!"), "should display as n!: {s}");
}

#[test]
fn binomial_display() {
    let n = symplex::default_context().symbol("n");
    let k = symplex::default_context().symbol("k");
    let expr = n.binomial(&k);
    let s = format!("{expr}");
    assert!(
        s.contains("C(") || s.contains("n") && s.contains("k"),
        "got: {s}"
    );
}
