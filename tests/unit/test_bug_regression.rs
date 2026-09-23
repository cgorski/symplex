//! Regression test for the proptest-discovered self-subtraction bug.
//!
//! The minimal failing input was:
//!   Mul(Neg(Int(1)), Add(Sym(0), Int(-1)))
//! which builds as (-1) * (-1 + a), and expr - expr did NOT produce zero.

use symplex::prelude::*;

#[test]
fn regression_mul_neg_one_times_add() {
    let ctx = Context::new();
    let a = ctx.symbol("a");

    // Build: (-1) * (-1 + a)
    let neg1 = ctx.int(-1);
    let sum = &a + &neg1; // canonicalizes to -1 + a
    let expr = &neg1 * &sum; // Mul(-1, Add(-1, a))

    eprintln!("expr = {expr}");

    // Self-subtraction must be zero
    let result = &expr - &expr;
    eprintln!("result = {result}");
    assert!(
        result.is_zero_structural(),
        "expr - expr should be structural zero, got: {result}"
    );
}

#[test]
fn regression_neg_of_product() {
    let ctx = Context::new();
    let a = ctx.symbol("a");

    let neg1 = ctx.int(-1);
    let sum = &a + &neg1;
    let expr = &neg1 * &sum;

    eprintln!("expr = {expr}");

    let neg_expr = -&expr;
    eprintln!("neg(expr) = {neg_expr}");

    // expr + neg(expr) must be zero
    let result = &expr + &neg_expr;
    eprintln!("expr + neg(expr) = {result}");
    assert!(
        result.is_zero_structural(),
        "expr + neg(expr) should be structural zero, got: {result}"
    );
}

#[test]
fn regression_simple_add_self_sub() {
    // Simpler case: (-1 + a) - (-1 + a) = 0
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let expr = &a + &ctx.int(-1); // -1 + a

    eprintln!("expr = {expr}");

    let result = &expr - &expr;
    eprintln!("result = {result}");
    assert!(
        result.is_zero_structural(),
        "(-1 + a) - (-1 + a) should be zero, got: {result}"
    );
}

#[test]
fn regression_neg_distributes_over_add() {
    // neg(-1 + a) should distribute to 1 + (-a) = 1 - a
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let expr = &a + &ctx.int(-1); // -1 + a
    let neg_expr = -&expr;

    eprintln!("expr = {expr}");
    eprintln!("neg(expr) = {neg_expr}");

    // neg should distribute: -(- 1 + a) = 1 - a = 1 + (-1)*a
    // Display should NOT contain a nested Add in parens like "-(- 1 + a)"
    let s = format!("{neg_expr}");
    assert!(
        !s.contains("(-"),
        "neg should distribute over Add, not wrap it: got {s}"
    );
}

#[test]
fn regression_as_coeff_term_roundtrip() {
    // For any expression, as_coeff_term followed by make_coeff_term
    // should return the original ExprId.
    let ctx = Context::new();
    let a = ctx.symbol("a");

    // Test with a simple Mul
    let expr = &a * 3; // 3*a
    let _expr_display = format!("{expr}");

    // Build it a different way — should be same ExprId
    let three = ctx.int(3);
    let expr2 = &three * &a;
    assert_eq!(
        format!("{expr}"),
        format!("{expr2}"),
        "3*a built two ways should match"
    );
    assert_eq!(expr, expr2, "3*a built two ways should be same ExprId");

    // Test with Mul of multiple factors
    let b = ctx.symbol("b");
    let expr3 = &a * &b * 2; // 2*a*b
    let expr3_display = format!("{expr3}");
    eprintln!("2*a*b = {expr3_display}");

    // Subtracting from itself must give zero
    let zero = &expr3 - &expr3;
    assert!(
        zero.is_zero_structural(),
        "2*a*b - 2*a*b should be zero, got: {zero}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Regressions from Phase 5 bug audit
// ═══════════════════════════════════════════════════════════════════════════

/// Regression: smart_simplify dropped the GCD factor from factor_terms.
/// Input: 6x + 12. Strategy 4 extracted (6, x+2), simplified x+2,
/// and returned it — losing the factor of 6.
#[test]
fn regression_smart_simplify_gcd_dropped() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x * 6) + 12;
    let result = expr.simplify();
    // Must be mathematically equal to 6x + 12
    let point = ctx.rational(7, 10);
    let val_orig = expr.subs(&x, &point).eval_f64().unwrap();
    let val_result = result.subs(&x, &point).eval_f64().unwrap();
    assert!(
        (val_orig - val_result).abs() < 1e-10,
        "smart_simplify(6x + 12) changed the value: {val_orig} vs {val_result}"
    );
}

/// Regression: smart_simplify with GCD should still simplify inner expression.
/// Input: 2*sin(x)^2 + 2*cos(x)^2 should become 2 (not stay as-is).
#[test]
fn regression_smart_simplify_gcd_with_pythagorean() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x.sin().powi(2) * 2) + &(&x.cos().powi(2) * 2);
    let result = expr.simplify();
    let result_str = format!("{result}");
    assert_eq!(
        result_str, "2",
        "2sin²+2cos² should smart_simplify to 2, got: {result_str}"
    );
}

/// Regression: integration by-parts caused stack overflow on x·ln(x).
/// The by-parts heuristic tried u=x, dv=ln(x) first, leading to
/// ∫ v·du = ∫ (x·ln(x) - x) dx which contains the original integral.
#[test]
fn regression_by_parts_x_ln_x_no_crash() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * &x.ln();
    let result = expr.integrate(&x);
    // Should produce a result (not crash, not unevaluated)
    let result_str = format!("{result}");
    assert!(
        !result_str.contains("Integral"),
        "∫ x·ln(x) dx should be integrable, got: {result_str}"
    );
    // Verify by differentiation
    let deriv = result.diff(&x);
    let point = ctx.int(2);
    let val_orig = expr.subs(&x, &point).eval_f64().unwrap();
    let val_deriv = deriv.subs(&x, &point).eval_f64().unwrap();
    assert!(
        (val_orig - val_deriv).abs() < 1e-8,
        "d/dx(∫ x·ln(x) dx) should equal x·ln(x) at x=2: {val_orig} vs {val_deriv}"
    );
}

/// Regression: pow_pow fired without checking if exponents are integers.
/// ((-1)^2)^(1/2) should be 1, not -1.
#[test]
fn regression_pow_pow_negative_base() {
    let ctx = Context::new();
    // 0.23: the identity needs a real argument (a symbol without assumptions may be complex).
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    // (x^2)^(1/2) should give |x| via sqrt_sq, not x via pow_pow
    let half = ctx.rational(1, 2);
    let expr = x.powi(2).pow(&half);
    let result = expr.simplify();
    let result_str = format!("{result}");
    assert_eq!(
        result_str, "abs(x)",
        "(x^2)^(1/2) should simplify to abs(x), got: {result_str}"
    );
}

/// Regression: asin(sin(x)) was simplified to x for symbolic x,
/// which is wrong when x ∉ [-π/2, π/2].
#[test]
fn regression_asin_sin_symbolic_not_simplified() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().asin();
    let result = expr.simplify();
    let result_str = format!("{result}");
    assert_eq!(
        result_str, "asin(sin(x))",
        "asin(sin(x)) should stay for symbolic x, got: {result_str}"
    );
}

/// Regression: acosh(cosh(x)) was simplified to x instead of |x|.
/// cosh is even, so acosh(cosh(-5)) = 5, not -5.
#[test]
fn regression_acosh_cosh_gives_abs() {
    let ctx = Context::new();
    // 0.23: the identity needs a real argument (a symbol without assumptions may be complex).
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let expr = x.cosh().acosh();
    let result = expr.simplify();
    let result_str = format!("{result}");
    assert_eq!(
        result_str, "abs(x)",
        "acosh(cosh(x)) should give |x|, got: {result_str}"
    );
}
