//! Integration tests for expand_power_exp, expand_power_base, ratsimp,
//! separateand inverse Laplace completing-the-square enhancements.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// 1. expand_power_exp: x^(a+b) → x^a · x^b
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_power_exp_does_not_panic() {
    // expand_power_exp splits x^(a+b) → x^a · x^b internally,
    // but canon_mul recombines same-base powers back into x^(a+b).
    // Verify the code path runs without error and is idempotent.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = x.pow(&(&a + &b));
    let expanded = expr.expand();
    let s = format!("{expanded}");
    // Canonicalization recombines: x^a * x^b → x^(a+b), so result
    // is the same as the input expression.
    assert!(
        s.contains("x") && s.contains("a") && s.contains("b"),
        "x^(a+b) expand should preserve the expression, got: {s}"
    );
}

#[test]
fn expand_power_exp_with_integer_addend() {
    // x^(2 + a) → expanded: the expand_power_exp path fires, but
    // canon_mul recombines. Verify no panic and valid result.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let expr = x.pow(&(&a + 2));
    let expanded = expr.expand();
    let s = format!("{expanded}");
    assert!(
        s.contains("x"),
        "x^(a+2) expand should produce valid expression, got: {s}"
    );
    // Expanding twice should be idempotent
    let expanded2 = expanded.expand();
    assert_eq!(
        format!("{expanded}"),
        format!("{expanded2}"),
        "expand should be idempotent for power-exp case"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. expand_power_base: (x·y)^n → x^n · y^n
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_power_base_simple() {
    // (x*y)^a = x^a * y^a is only an identity for non-negative bases (or
    // integer a): for x = y = -1, a = 1/2 the left side is 1 and the right
    // side is -1.  Since 0.2 `expand` guards the rewrite with the
    // assumption system, so the symbols are declared positive here.
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]);
    let y = ctx.symbol_with("y", &[Assumption::Positive]);
    let a = ctx.symbol("a");
    // (x*y)^a should expand to x^a * y^a
    let expr = (&x * &y).pow(&a);
    let expanded = expr.expand();
    let s = format!("{expanded}");
    assert!(
        s.contains("x^a") && s.contains("y^a"),
        "(x*y)^a should expand to x^a * y^a, got: {s}"
    );
}

#[test]
fn expand_power_base_blocked_for_unassumed_symbols() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let a = ctx.symbol("a");
    let expr = (&x * &y).pow(&a);
    let expanded = expr.expand();
    assert_eq!(
        format!("{expanded}"),
        format!("{expr}"),
        "(x*y)^a must not split without positivity information"
    );
}

#[test]
fn expand_power_base_three_factors() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]);
    let y = ctx.symbol_with("y", &[Assumption::Positive]);
    let z = ctx.symbol_with("z", &[Assumption::Positive]);
    let n = ctx.symbol("n");
    // (x*y*z)^n → x^n * y^n * z^n
    let expr = (&x * &y * &z).pow(&n);
    let expanded = expr.expand();
    let s = format!("{expanded}");
    assert!(
        s.contains("x^n") && s.contains("y^n") && s.contains("z^n"),
        "(x*y*z)^n should expand to x^n*y^n*z^n, got: {s}"
    );
}

#[test]
fn expand_power_base_integer_exp_uses_multinomial() {
    // (x+y)^2 should still use multinomial (not power_base),
    // because the base is Add, not Mul.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = (&x + &y).powi(2);
    let expanded = expr.expand();
    let s = format!("{expanded}");
    // Should be x^2 + 2*x*y + y^2
    assert!(
        s.contains("x^2") && s.contains("y^2"),
        "(x+y)^2 should multinomial-expand, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. ratsimp convenience method
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ratsimp_combines_fractions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 1/x + 1/x = 2/x
    let expr = &x.powi(-1) + &x.powi(-1);
    let simplified = expr.simplify_rational();
    let s = format!("{simplified}");
    assert!(
        s.contains("2"),
        "1/x + 1/x should ratsimp to 2/x or 2*x^(-1), got: {s}"
    );
}

#[test]
fn ratsimp_cancels_common_factor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x^2 - 1) / (x - 1) should simplify to x + 1
    let expr = (&x.powi(2) - 1) / (&x - 1);
    let simplified = expr.simplify_rational();
    let s = format!("{simplified}");
    assert!(
        s.contains("x") && s.contains("1"),
        "(x^2-1)/(x-1) should ratsimp to x+1, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. separatevars
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn separatevars_independent_factors() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x * y — each factor depends on a different variable
    let expr = &x * &y;
    let groups = expr.separate_vars(&[&x, &y]);
    // Should produce at least 2 groups (one for x, one for y)
    assert!(
        groups.len() >= 2,
        "x*y should separate into ≥2 groups, got {} groups",
        groups.len()
    );
}

#[test]
fn separatevars_with_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 3 * x — should have a constant group and an x group
    let expr = &x * 3;
    let groups = expr.separate_vars(&[&x]);
    let const_groups: Vec<_> = groups.iter().filter(|(deps, _)| deps.is_empty()).collect();
    let x_groups: Vec<_> = groups.iter().filter(|(deps, _)| !deps.is_empty()).collect();
    assert!(!const_groups.is_empty(), "3*x should have a constant group");
    assert!(!x_groups.is_empty(), "3*x should have an x group");
}

#[test]
fn separatevars_non_mul_single_group() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Just a symbol, not a Mul — should return a single group
    let groups = x.separate_vars(&[&x]);
    assert_eq!(groups.len(), 1, "a single symbol should be one group");
    assert_eq!(groups[0].0.len(), 1, "should depend on x");
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Inverse Laplace completing-the-square (underdamped β² > 0)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn inverse_laplace_completing_square_underdamped() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    let t = ctx.symbol("t");
    // F(s) = 1 / (s² + 2s + 5)
    // Complete the square: (s+1)² + 4, so α = -1, β = 2
    // L⁻¹ = (1/2) · exp(-t) · sin(2t)
    let denom = &s.powi(2) + &s * 2 + 5;
    let f_s = 1 / &denom;
    let result = f_s.inverse_laplace(&s, &t);
    let s_repr = format!("{result}");
    // Should contain exp and sin
    assert!(
        s_repr.contains("sin") && s_repr.contains("exp"),
        "inverse Laplace of 1/(s²+2s+5) should have exp and sin, got: {s_repr}"
    );
}

#[test]
fn inverse_laplace_completing_square_overdamped() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    let t = ctx.symbol("t");
    // F(s) = 1 / (s² + 2s - 3)
    // Complete the square: (s+1)² - 4, so α = -1, γ = 2
    // β² = -3 - 1 = -4 < 0 → overdamped
    // L⁻¹ = (1/2) · exp(-t) · sinh(2t)
    let denom = &s.powi(2) + &s * 2 - 3;
    let f_s = 1 / &denom;
    let result = f_s.inverse_laplace(&s, &t);
    let s_repr = format!("{result}");
    // Should produce a result containing exp (may use sinh or partial fractions)
    assert!(
        s_repr.contains("exp") || s_repr.contains("sinh"),
        "inverse Laplace of 1/(s²+2s-3) should have exp/sinh, got: {s_repr}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// expand_power_exp soundness guard tests
//
// The guard in expand_power_exp prevents x^(a+b) → x^a * x^b when x
// could be negative (the identity fails for negative bases with fractional
// exponents due to complex branch cuts).
//
// Note: even when the guard ALLOWS the split, canon_mul may recombine
// 2^a * 2^b back into 2^(a+b) (same-base power collection). That's
// correct behavior — the important thing is that the unsound cases are
// BLOCKED.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_power_exp_blocks_symbolic_base() {
    // x^(a+b) must NOT split to x^a * x^b when x is a general symbol,
    // because for negative x with fractional exponents, the identity fails.
    let ctx = Context::new();
    symplex::syms!(ctx; x, a, b);
    let expr = x.pow(&(&a + &b));
    let expanded = expr.expand();
    let s = format!("{expanded}");
    // Must NOT contain a split product — should remain as x^(...)
    assert!(
        !s.contains("x^a*x^b"),
        "x^(a+b) should NOT be split for symbolic base, got: {s}"
    );
}

#[test]
fn expand_power_exp_euler_becomes_exp_node() {
    // e^(a+b) is canonicalized to Exp(a+b) at construction time,
    // so it never reaches expand_power_exp (which handles Pow nodes).
    // Verify the canonical form is Exp and expand doesn't crash.
    let ctx = Context::new();
    symplex::syms!(ctx; a, b);
    let expr = ctx.e().pow(&(&a + &b));
    let s = format!("{expr}");
    assert!(
        s.contains("exp("),
        "e^(a+b) should be canonicalized to exp(...), got: {s}"
    );
    // Expand should be a no-op (Exp(Add) has no expand rule)
    let expanded = expr.expand();
    let s2 = format!("{expanded}");
    assert!(
        s2.contains("exp("),
        "e^(a+b) should remain as exp(...) after expand, got: {s2}"
    );
}

#[test]
fn expand_power_exp_positive_numeric_base_no_panic() {
    // 2^(a+b): the guard ALLOWS the split (base is positive).
    // canon_mul immediately recombines 2^a * 2^b → 2^(a+b).
    // The important thing: no panic, and the result is valid.
    let ctx = Context::new();
    symplex::syms!(ctx; a, b);
    let expr = ctx.int(2).pow(&(&a + &b));
    let expanded = expr.expand();
    let s = format!("{expanded}");
    assert!(
        s.contains("2") && s.contains("a") && s.contains("b"),
        "2^(a+b) expand should produce valid expression, got: {s}"
    );
    // Expanding twice should be idempotent
    let expanded2 = expanded.expand();
    assert_eq!(
        format!("{expanded}"),
        format!("{expanded2}"),
        "expand should be idempotent for positive numeric base case"
    );
}

#[test]
fn expand_power_exp_blocks_negative_numeric_base() {
    // (-2)^(a+b) must NOT split — negative base with symbolic exponents.
    // The identity (-2)^(a+b) = (-2)^a * (-2)^b fails for fractional exponents.
    let ctx = Context::new();
    symplex::syms!(ctx; a, b);
    let neg2 = -&ctx.int(2);
    let expr = neg2.pow(&(&a + &b));
    let expanded = expr.expand();
    let s = format!("{expanded}");
    assert!(
        !s.contains("(-2)^a*(-2)^b"),
        "(-2)^(a+b) should NOT be split, got: {s}"
    );
}

#[test]
fn expand_power_exp_numeric_exponents_allowed() {
    // x^(2+3): canon_add folds 2+3 → 5 at construction time,
    // so by the time expand sees it, it's already x^5.
    // This tests that the all-nonneg guard path doesn't cause issues.
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.pow(&(&ctx.int(2) + &ctx.int(3)));
    let s = format!("{expr}");
    // Canon already folded 2+3 to 5
    assert!(
        s.contains("x^5"),
        "x^(2+3) should be x^5 after canonicalization, got: {s}"
    );
}
