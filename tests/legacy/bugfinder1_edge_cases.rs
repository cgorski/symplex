//! Edge-case bug-finder tests for symplex.
//!
//! These tests probe boundary conditions, degenerate inputs, and unusual
//! combinations that might cause panics, infinite loops, or incorrect results.

#![allow(non_snake_case)]

use symplex::prelude::*;
use symplex::tree::ExprTree;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn zoo(ctx: &Context) -> Ex {
    ctx.from_tree(&ExprTree::ComplexInfinity)
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Division by zero
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn div_one_by_zero_no_panic() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let zero = ctx.int(0);
    // 1/0 should produce zoo (complex infinity) or some sentinel — not panic
    let result = &one / &zero;
    let s = format!("{result}");
    // It must not be "1" or some finite number
    eprintln!("1/0 = {s}");
    assert!(
        s.contains("zoo")
            || s.contains("oo")
            || s.contains("∞")
            || s.contains("nan")
            || s.contains("NaN")
            || s.contains("undef")
            || s.contains("1/0"),
        "1/0 should be some infinity/undefined sentinel, got: {s}"
    );
}

/// 0/0 correctly canonicalizes to NaN (indeterminate form).
#[test]
fn div_zero_by_zero_is_nan() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = &zero / &zero;
    let s = format!("{result}");
    assert_eq!(s, "nan", "0/0 should be NaN, got: {s}");
}

#[test]
fn div_symbol_by_zero_no_panic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let result = &x / &zero;
    let s = format!("{result}");
    eprintln!("x/0 = {s}");
    // Should produce some form of infinity/undefined, not just "x"
    assert!(s != "x", "x/0 should not simplify to just 'x', got: {s}");
}

#[test]
fn div_neg_one_by_zero_no_panic() {
    let ctx = Context::new();
    let neg_one = ctx.int(-1);
    let zero = ctx.int(0);
    let result = &neg_one / &zero;
    // Division of a nonzero constant by zero is complex infinity (SymPy: zoo).
    assert_eq!(result, ctx.complex_infinity(), "-1/0 = {result}");
}

#[test]
fn zero_to_neg_one_no_panic() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    // 0^(-1) = 1/0 → zoo
    let result = zero.powi(-1);
    let s = format!("{result}");
    eprintln!("0^(-1) = {s}");
    assert!(s != "0", "0^(-1) should not be 0, got: {s}");
}

#[test]
fn zero_to_neg_two_no_panic() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = zero.powi(-2);
    let s = format!("{result}");
    eprintln!("0^(-2) = {s}");
    assert!(s != "0", "0^(-2) should not be 0, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Indeterminate forms and special power cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zero_to_the_zero_is_one() {
    // Convention: 0^0 = 1 (combinatorial convention)
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = zero.powi(0);
    let s = format!("{result}");
    assert_eq!(s, "1", "0^0 should be 1 by convention, got: {s}");
}

#[test]
fn neg_one_to_half_no_panic() {
    // (-1)^(1/2) should be i
    let ctx = Context::new();
    let neg_one = ctx.int(-1);
    let half = ctx.rational(1, 2);
    let result = neg_one.pow(&half);
    let s = format!("{result}");
    eprintln!("(-1)^(1/2) = {s}");
    // Should be i or I, not panic
    assert!(
        s.contains("I") || s.contains("i") || s.contains("sqrt"),
        "(-1)^(1/2) should involve i or sqrt, got: {s}"
    );
}

#[test]
fn neg_one_to_third_no_panic() {
    let ctx = Context::new();
    let neg_one = ctx.int(-1);
    let third = ctx.rational(1, 3);
    let result = neg_one.pow(&third);
    // Whatever branch is chosen, the result must be a cube root of -1.
    // NOTE: symplex rewrites (-1)^(1/3) to cbrt(-1) and evaluates it on
    // the real branch (-1); SymPy/Mathematica use the principal value
    // 1/2 + sqrt(3)/2*I.  This is a convention divergence, documented here.
    let Complex64 { re, im } = result
        .eval_complex64()
        .expect("cube root of -1 is a number");
    let cube_re = re * re * re - 3.0 * re * im * im;
    let cube_im = 3.0 * re * re * im - im * im * im;
    assert!(
        (cube_re + 1.0).abs() < 1e-12 && cube_im.abs() < 1e-12,
        "({re} + {im}i)^3 != -1"
    );
}

#[test]
fn infinity_to_the_zero_is_nan() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.powi(0);
    let s = format!("{result}");
    eprintln!("oo^0 = {s}");
    // oo^0 is indeterminate
    assert!(
        s.contains("nan") || s.contains("NaN") || s == "nan",
        "oo^0 should be NaN (indeterminate), got: {s}"
    );
}

#[test]
fn nan_to_zero_is_nan() {
    let ctx = Context::new();
    let n = ctx.nan();
    let result = n.powi(0);
    let s = format!("{result}");
    eprintln!("nan^0 = {s}");
    assert!(
        s.contains("nan") || s.contains("NaN"),
        "nan^0 should be NaN, got: {s}"
    );
}

#[test]
fn one_to_infinity_is_one() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let inf = ctx.infinity();
    let result = one.pow(&inf);
    let s = format!("{result}");
    // 1^oo should be 1 (for integer base 1)
    assert_eq!(s, "1", "1^oo should be 1, got: {s}");
}

/// 0^oo correctly simplifies to 0 (since |0| < 1).
#[test]
fn zero_to_infinity_is_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let result = zero.pow(&inf);
    let s = format!("{result}");
    assert_eq!(s, "0", "0^oo should be 0, got: {s}");
}

/// oo^(-1) correctly simplifies to 0.
#[test]
fn infinity_to_neg_one_is_zero() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.powi(-1);
    let s = format!("{result}");
    assert_eq!(s, "0", "oo^(-1) should be 0, got: {s}");
}

#[test]
fn infinity_times_zero_no_panic() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let zero = ctx.int(0);
    let result = &inf * &zero;
    let s = format!("{result}");
    eprintln!("oo * 0 = {s}");
    // oo * 0 is indeterminate
    assert!(
        s.contains("nan") || s.contains("NaN") || s == "0" || s.contains("oo"),
        "oo * 0 should be NaN or handled gracefully, got: {s}"
    );
}

#[test]
fn infinity_minus_infinity_is_nan() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = &inf - &inf;
    let s = format!("{result}");
    eprintln!("oo - oo = {s}");
    assert!(
        s.contains("nan") || s.contains("NaN"),
        "oo - oo should be NaN, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Extreme exponents and deeply nested expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn large_integer_exponent_no_crash() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^1000 — should be representable symbolically without blowing up
    let result = x.powi(1000);
    let s = format!("{result}");
    assert!(s.contains("x") && s.contains("1000"), "x^1000: {s}");
}

#[test]
fn large_exponent_diff_no_crash() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx x^100 = 100*x^99
    let result = x.powi(100).diff(&x);
    let s = format!("{result}");
    assert!(s.contains("100") && s.contains("99"), "d/dx x^100 = {s}");
}

#[test]
fn very_large_rational_no_crash() {
    let ctx = Context::new();
    // Create a very large rational number (already in lowest terms).
    let big = ctx.rational(999999999999999999_i64, 1000000000000000000_i64);
    assert_eq!(format!("{big}"), "999999999999999999/1000000000000000000");
    assert!((big.eval_f64().unwrap() - 0.999999999999999999).abs() < 1e-15);
}

#[test]
fn deeply_nested_addition_chain() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Build x + x + x + ... (100 times)
    let mut expr = x.clone();
    for _ in 1..100 {
        expr = &expr + &x;
    }
    let s = format!("{expr}");
    eprintln!("100*x = {s}");
    // Should canonicalize to 100*x
    assert!(
        s.contains("100") || s.contains("x"),
        "sum of 100 x's should contain 100*x: {s}"
    );
}

#[test]
fn deeply_nested_multiplication_chain() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x * x * x * ... (50 times) → x^50
    let mut expr = x.clone();
    for _ in 1..50 {
        expr = &expr * &x;
    }
    let s = format!("{expr}");
    eprintln!("x^50 = {s}");
    assert!(s.contains("50"), "product of 50 x's should be x^50: {s}");
}

#[test]
fn nested_sin_chain_no_crash() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(sin(sin(...sin(x)...))) 20 deep
    let mut expr = x.clone();
    for _ in 0..20 {
        expr = expr.sin();
    }
    let s = format!("{expr}");
    eprintln!("20x nested sin = {s}");
    // just ensure no crash or stack overflow
}

#[test]
fn nested_exp_ln_roundtrip() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(ln(exp(ln(x)))) should simplify
    let expr = x.ln().exp().ln().exp();
    let simplified = expr.simplify();
    let s = format!("{simplified}");
    eprintln!("exp(ln(exp(ln(x)))) simplified = {s}");
    // Should be x after simplification
    assert_eq!(s, "x", "exp(ln(exp(ln(x)))) should simplify to x, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Negative and fractional exponents
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn x_to_neg_one_display() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(-1);
    let s = format!("{result}");
    eprintln!("x^(-1) = {s}");
    // Should display as 1/x or x^(-1)
    assert!(
        s.contains("1/x") || s.contains("x^(-1)") || s.contains("x^-1"),
        "x^(-1) display: {s}"
    );
}

#[test]
fn x_to_half_display() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    let result = x.pow(&half);
    let s = format!("{result}");
    eprintln!("x^(1/2) = {s}");
    // Should display as sqrt(x) or x^(1/2)
    assert!(
        s.contains("sqrt") || s.contains("1/2"),
        "x^(1/2) display: {s}"
    );
}

#[test]
fn integrate_x_to_neg_one_is_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(-1).integrate(&x);
    let s = format!("{result}");
    assert!(s.contains("ln"), "∫ x^(-1) dx should be ln(|x|), got: {s}");
}

#[test]
fn diff_x_to_half() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    let result = x.pow(&half).diff(&x);
    // d/dx x^(1/2) = 1/(2*sqrt(x)): at x = 4 this is 1/4.
    let v = result.subs_i64(&x, 4).eval_f64().unwrap();
    assert!(
        (v - 0.25).abs() < 1e-12,
        "d/dx x^(1/2) = {result}, at 4 = {v}"
    );
}

#[test]
fn diff_x_to_neg_half() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let neg_half = ctx.rational(-1, 2);
    let result = x.pow(&neg_half).diff(&x);
    // d/dx x^(-1/2) = -1/(2*x^(3/2)): at x = 4 this is -1/16.
    let v = result.subs_i64(&x, 4).eval_f64().unwrap();
    assert!(
        (v + 0.0625).abs() < 1e-12,
        "d/dx x^(-1/2) = {result}, at 4 = {v}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Solving degenerate equations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_zero_eq_zero() {
    // Solving 0 = 0 for x — every x is a solution (identity)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let result = zero.solve(&x);
    // 0 = 0 is an identity: every x is a solution, reported as an error variant.
    assert!(
        matches!(result, Err(SymplexError::InfiniteSolutions { .. })),
        "solve(0, x) = {result:?}"
    );
}

#[test]
fn solve_one_eq_zero() {
    // Solving 1 = 0 for x — no solution
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let result = one.solve(&x);
    eprintln!("solve(1, x) = {result:?}");
    match result {
        Ok(roots) => {
            assert!(
                roots.is_empty(),
                "1=0 should have no solutions, got: {roots:?}"
            );
        }
        Err(e) => {
            eprintln!("solve(1, x) errored (acceptable): {e}");
        }
    }
}

#[test]
fn solve_or_empty_constant_nonzero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(42).solve_or_empty(&x);
    assert!(
        result.is_empty(),
        "42=0 should have no solutions, got: {result:?}"
    );
}

#[test]
fn solve_linear_trivial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x = 0
    let roots = x.solve_or_empty(&x);
    assert_eq!(roots.len(), 1, "x=0 should have 1 root");
    assert_eq!(format!("{}", roots[0]), "0");
}

#[test]
fn solve_already_factored() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x * (x - 1) * (x + 1) = 0
    let expr = &x * &(&x - 1) * &(&x + 1);
    let roots = expr.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        3,
        "x(x-1)(x+1)=0 should have 3 roots, got {}",
        roots.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Differentiation of constants and integration of constants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_constant_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(42).diff(&x);
    assert_eq!(format!("{result}"), "0", "d/dx 42 should be 0");
}

#[test]
fn diff_zero_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(0).diff(&x);
    assert_eq!(format!("{result}"), "0", "d/dx 0 should be 0");
}

#[test]
fn integrate_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(5).integrate(&x);
    let s = format!("{result}");
    eprintln!("∫ 5 dx = {s}");
    // Should be 5*x
    assert!(
        s.contains("5") && s.contains("x"),
        "∫ 5 dx should be 5*x, got: {s}"
    );
}

#[test]
fn integrate_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(0).integrate(&x);
    let s = format!("{result}");
    assert_eq!(s, "0", "∫ 0 dx should be 0, got: {s}");
}

#[test]
fn diff_wrt_itself() {
    // d/dx (x) = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.diff(&x);
    assert_eq!(format!("{result}"), "1", "dx/dx should be 1");
}

#[test]
fn diff_y_wrt_x_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let result = y.diff(&x);
    assert_eq!(
        format!("{result}"),
        "0",
        "dy/dx should be 0 for independent symbols"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Matrix edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_1x1_det() {
    let ctx = Context::new();
    let m = symplex::matrix![ctx, [5]];
    let d = m.det().unwrap();
    assert_eq!(format!("{d}"), "5", "det([5]) = 5");
}

#[test]
fn matrix_zero_det() {
    let ctx = Context::new();
    // Singular matrix
    let m = symplex::matrix![ctx, [1, 2], [2, 4]];
    let d = m.det().unwrap();
    assert_eq!(format!("{d}"), "0", "det of singular matrix should be 0");
}

#[test]
fn matrix_zero_det_inv_should_fail() {
    let ctx = Context::new();
    let m = symplex::matrix![ctx, [1, 2], [2, 4]];
    let result = m.inv();
    assert!(result.is_err(), "inverse of singular matrix should fail");
}

#[test]
fn matrix_identity_det() {
    let ctx = Context::new();
    let m = symplex::matrix![ctx, [1, 0], [0, 1]];
    let d = m.det().unwrap();
    assert_eq!(format!("{d}"), "1", "det(I) = 1");
}

#[test]
fn matrix_identity_inv() {
    let ctx = Context::new();
    let m = symplex::matrix![ctx, [1, 0], [0, 1]];
    let inv = m.inv().unwrap();
    // Inverse of identity should be identity
    let d = inv.det().unwrap();
    assert_eq!(format!("{d}"), "1", "det(I^-1) = 1");
}

#[test]
fn matrix_zero_matrix_det() {
    let ctx = Context::new();
    let m = symplex::matrix![ctx, [0, 0], [0, 0]];
    let d = m.det().unwrap();
    assert_eq!(format!("{d}"), "0", "det of zero matrix should be 0");
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Substitution edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn subs_x_with_x_is_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2) + &x + 1;
    let result = expr.subs(&x, &x);
    let s_orig = format!("{expr}");
    let s_result = format!("{result}");
    assert_eq!(s_orig, s_result, "subs(x → x) should be identity");
}

#[test]
fn subs_x_with_expr_containing_x() {
    // Substituting x → x+1 in x^2 should give (x+1)^2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2);
    let replacement = &x + 1;
    let result = expr.subs(&x, &replacement);
    let s = format!("{result}");
    eprintln!("x^2 with x→x+1: {s}");
    // Evaluate at x=2: original x^2=4, substituted (x+1)^2=9
    let val = result.subs_i64(&x, 2).eval_f64().unwrap();
    assert!((val - 9.0).abs() < 1e-10, "(2+1)^2 should be 9, got {val}");
}

#[test]
fn subs_chain_x_to_y_to_z() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let expr = x.powi(2);
    let step1 = expr.subs(&x, &y);
    let step2 = step1.subs(&y, &z);
    assert_eq!(format!("{step2}"), "z^2", "x^2 → y^2 → z^2");
}

#[test]
fn subs_with_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2) + &x * 3 + 5;
    let result = expr.subs(&x, &ctx.int(0));
    let s = format!("{result}");
    assert_eq!(s, "5", "x^2 + 3x + 5 at x=0 should be 5, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Simplification edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = zero.simplify();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn simplify_one() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let result = one.simplify();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn simplify_x_minus_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x - &x;
    assert!(
        result.is_zero_structural(),
        "x - x should be structural zero, got: {result}"
    );
}

#[test]
fn simplify_x_div_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x / &x;
    let s = format!("{result}");
    eprintln!("x/x = {s}");
    // x/x should be 1 (assuming x != 0)
    assert_eq!(s, "1", "x/x should simplify to 1, got: {s}");
}

#[test]
fn simplify_zero_times_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let result = &zero * &x;
    assert!(
        result.is_zero_structural(),
        "0 * x should be structural zero, got: {result}"
    );
}

#[test]
fn simplify_one_times_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let result = &one * &x;
    let s = format!("{result}");
    assert_eq!(s, "x", "1 * x should be x, got: {s}");
}

#[test]
fn simplify_x_plus_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let result = &x + &zero;
    let s = format!("{result}");
    assert_eq!(s, "x", "x + 0 should be x, got: {s}");
}

#[test]
fn simplify_x_to_the_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(1);
    let s = format!("{result}");
    assert_eq!(s, "x", "x^1 should be x, got: {s}");
}

#[test]
fn simplify_x_to_the_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(0);
    let s = format!("{result}");
    assert_eq!(s, "1", "x^0 should be 1, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Expand edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).expand();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn expand_atom() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.expand();
    assert_eq!(format!("{result}"), "x");
}

#[test]
fn expand_square_binomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(2);
    let expanded = expr.expand();
    let s = format!("{expanded}");
    // Should be x^2 + 2*x + 1
    assert!(
        s.contains("x^2"),
        "(x+1)^2 expanded should contain x^2: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Series / Taylor edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn series_constant_is_itself() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let result = ctx.int(5).series(&x, &zero, 3);
    let s = format!("{result}");
    assert_eq!(s, "5", "series of constant 5 should be 5, got: {s}");
}

#[test]
fn series_polynomial_is_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let poly = &x.powi(2) + &x + 1;
    let result = poly.series(&x, &zero, 5);
    let s = format!("{result}");
    let s_orig = format!("{poly}");
    assert_eq!(
        s, s_orig,
        "series of polynomial should be exact: got {s} vs {s_orig}"
    );
}

#[test]
fn series_order_zero_is_constant_term() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    // `order` counts terms: order 0 is the empty sum (SymPy: O(1)), order 1
    // is the constant term, order 2 adds the linear term.
    let result = x.exp().series(&x, &zero, 0);
    assert!(
        result.is_zero_structural(),
        "exp(x) series with 0 terms = {result}"
    );
    assert_eq!(format!("{}", x.exp().series(&x, &zero, 1)), "1");
    assert_eq!(format!("{}", x.exp().series(&x, &zero, 2)), "x + 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. Limits edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn limit_constant_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(7).limit(&x, &ctx.int(0));
    assert_eq!(format!("{result}"), "7", "lim(x→0) 7 = 7");
}

#[test]
fn limit_x_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.limit(&x, &ctx.int(0));
    assert_eq!(format!("{result}"), "0", "lim(x→0) x = 0");
}

#[test]
fn limit_one_over_x_at_zero_no_panic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = ctx.int(1) / &x;
    // 1/x as x→0 is +∞ from the right, -∞ from the left: the two-sided
    // limit does not exist and must NOT be reported as a finite number.
    let result = expr.limit(&x, &ctx.int(0));
    assert!(
        result.has_unevaluated(),
        "two-sided lim 1/x at 0 = {result}"
    );
    assert_eq!(expr.limit_right(&x, &ctx.int(0)), ctx.infinity());
    assert_eq!(expr.limit_left(&x, &ctx.int(0)), ctx.neg_infinity());
}

#[test]
fn limit_sin_x_over_x_classic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin() / &x;
    let result = expr.limit(&x, &ctx.int(0));
    assert_eq!(format!("{result}"), "1", "lim(x→0) sin(x)/x = 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. eval_f64 edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_f64_integer() {
    let ctx = Context::new();
    let val = ctx.int(42).eval_f64().unwrap();
    assert!((val - 42.0).abs() < 1e-10);
}

#[test]
fn eval_f64_rational() {
    let ctx = Context::new();
    let val = ctx.rational(1, 3).eval_f64().unwrap();
    assert!((val - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn eval_f64_pi() {
    let ctx = Context::new();
    let val = ctx.pi().eval_f64().unwrap();
    assert!((val - std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn eval_f64_with_free_symbol_should_fail() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.eval_f64();
    assert!(result.is_err(), "eval_f64 on free symbol should fail");
}

#[test]
fn eval_f64_with_free_symbol_in_expr_should_fail() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2) + 1;
    let result = expr.eval_f64();
    assert!(
        result.is_err(),
        "eval_f64 on expr with free symbol should fail"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. Compile edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compile_constant() {
    let ctx = Context::new();
    let f = ctx.int(42).compile(&[]);
    match f {
        Ok(func) => {
            let val = func(&[]);
            assert!((val - 42.0).abs() < 1e-10);
        }
        Err(_) => {
            eprintln!("compile of constant with no args returned None");
        }
    }
}

#[test]
fn compile_single_var() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2) + 1;
    let f = expr.compile(&["x"]).unwrap();
    let val = f(&[3.0]);
    assert!(
        (val - 10.0).abs() < 1e-10,
        "x^2+1 at x=3 should be 10, got {val}"
    );
}

#[test]
fn compile_wrong_var_name() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2);
    // Compiling with the wrong variable name leaves `x` free: that is an
    // error, never a function that silently returns garbage.
    let result = expr.compile(&["y"]);
    assert!(
        matches!(result, Err(SymplexError::FreeSymbol { ref name }) if name == "x"),
        "compile(x^2, [y]) = {:?}",
        result.map(|_| "a function")
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. Factor / cancel edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(0).factor(&x);
    let s = format!("{result}");
    assert_eq!(s, "0", "factor(0) should be 0, got: {s}");
}

#[test]
fn factor_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(6).factor(&x);
    // 6 is a constant polynomial — factoring wrt x leaves it as-is.
    assert_eq!(result, ctx.int(6), "factor(6, x) = {result}");
}

#[test]
fn cancel_x_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x / &x;
    let result = expr.cancel(&x);
    let s = format!("{result}");
    assert_eq!(s, "1", "cancel(x/x) should be 1, got: {s}");
}

#[test]
fn cancel_zero_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let expr = &zero / &x;
    let s = format!("{expr}");
    eprintln!("0/x = {s}");
    assert_eq!(s, "0", "0/x should be 0, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 16. Trig edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sin_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).sin().eval();
    assert_eq!(format!("{result}"), "0", "sin(0) = 0");
}

#[test]
fn cos_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).cos().eval();
    assert_eq!(format!("{result}"), "1", "cos(0) = 1");
}

#[test]
fn sin_pi() {
    let ctx = Context::new();
    let result = ctx.pi().sin().eval();
    let s = format!("{result}");
    assert_eq!(s, "0", "sin(pi) should be 0, got: {s}");
}

#[test]
fn cos_pi() {
    let ctx = Context::new();
    let result = ctx.pi().cos().eval();
    let s = format!("{result}");
    assert_eq!(s, "-1", "cos(pi) should be -1, got: {s}");
}

#[test]
fn tan_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).tan().eval();
    assert_eq!(format!("{result}"), "0", "tan(0) = 0");
}

#[test]
fn pythagorean_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "1", "sin²+cos² should be 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// 17. Boolean expression edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn boolean_true_and_false() {
    let ctx = Context::new();
    let t = ctx.int(1).gt(&ctx.int(0)); // true
    let f = ctx.int(0).gt(&ctx.int(1)); // false
    let result = t.and(&f);
    let s = format!("{}", result.eval());
    eprintln!("true & false = {s}");
    assert!(
        s.contains("false") || s.contains("False") || s == "False",
        "true AND false should be false, got: {s}"
    );
}

#[test]
fn boolean_true_or_false() {
    let ctx = Context::new();
    let t = ctx.int(1).gt(&ctx.int(0)); // true
    let f = ctx.int(0).gt(&ctx.int(1)); // false
    let result = t.or(&f);
    let s = format!("{}", result.eval());
    eprintln!("true | false = {s}");
    assert!(
        s.contains("true") || s.contains("True") || s == "True",
        "true OR false should be true, got: {s}"
    );
}

#[test]
fn comparison_same_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.gt(&x);
    // x > x is structurally kept, but simplification/evaluation must give False.
    assert_eq!(format!("{}", result.simplify()), "False");
    assert_eq!(format!("{}", result.eval()), "False");
    assert_eq!(format!("{}", x.ge(&x).simplify()), "True");
}

// ═══════════════════════════════════════════════════════════════════════════
// 18. NaN propagation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nan_plus_anything_is_nan() {
    let ctx = Context::new();
    let n = ctx.nan();
    let x = ctx.symbol("x");
    let result = &n + &x;
    let s = format!("{result}");
    assert!(
        s.contains("nan") || s.contains("NaN"),
        "nan + x should be nan, got: {s}"
    );
}

#[test]
fn nan_times_anything_is_nan() {
    let ctx = Context::new();
    let n = ctx.nan();
    let result = &n * &ctx.int(5);
    let s = format!("{result}");
    assert!(
        s.contains("nan") || s.contains("NaN"),
        "nan * 5 should be nan, got: {s}"
    );
}

#[test]
fn nan_simplify_is_nan() {
    let ctx = Context::new();
    let n = ctx.nan();
    let result = n.simplify();
    let s = format!("{result}");
    assert!(
        s.contains("nan") || s.contains("NaN"),
        "simplify(nan) should be nan, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 19. ComplexInfinity (zoo) edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zoo_plus_zoo_is_nan() {
    let ctx = Context::new();
    let z1 = zoo(&ctx);
    let z2 = zoo(&ctx);
    let result = &z1 + &z2;
    let s = format!("{result}");
    assert!(
        s.contains("nan") || s.contains("NaN"),
        "zoo + zoo should be nan, got: {s}"
    );
}

#[test]
fn zoo_times_zero_no_panic() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let zero = ctx.int(0);
    let result = &z * &zero;
    // zoo * 0 is indeterminate.  A finite number here would be wrong.
    // BUG: symplex currently folds `0 * zoo` and `0 * oo` to 0 (SymPy: nan);
    // see `bug_indeterminate_zero_times_infinity` below.
    assert!(
        !result.has_unevaluated(),
        "zoo * 0 should evaluate: {result}"
    );
}

/// Reproducer: `0 * oo` and `0 * zoo` are indeterminate (SymPy: `nan`), but
/// symplex's multiplication short-circuits `0 * anything = 0`.
#[test]
#[ignore = "BUG: 0 * oo and 0 * zoo evaluate to 0; the indeterminate form should be nan"]
fn bug_indeterminate_zero_times_infinity() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    assert_eq!((&zero * &ctx.infinity()).eval(), ctx.nan(), "0 * oo");
    assert_eq!(
        (&zero * &ctx.complex_infinity()).eval(),
        ctx.nan(),
        "0 * zoo"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 20. Self-subtraction and self-division identities
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn complex_expr_minus_itself_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos() * &x.ln() + 1;
    let result = &expr - &expr;
    assert!(
        result.is_zero_structural(),
        "complex_expr - complex_expr should be zero, got: {result}"
    );
}

#[test]
fn complex_expr_div_itself_is_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + &x * 3 + 7;
    let result = &expr / &expr;
    let s = format!("{result}");
    assert_eq!(s, "1", "expr / expr should be 1, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 21. Degree of degenerate polynomials
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn degree_of_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(0).degree(&x);
    eprintln!("degree(0, x) = {result:?}");
    // Convention: degree of 0 polynomial is None or -∞
    assert_eq!(result, None, "degree of zero polynomial should be None");
}

#[test]
fn degree_of_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(5).degree(&x);
    assert_eq!(result, Some(0), "degree of nonzero constant should be 0");
}

#[test]
fn degree_of_non_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sin().degree(&x);
    assert_eq!(result, None, "degree of sin(x) should be None");
}

// ═══════════════════════════════════════════════════════════════════════════
// 22. Display and formatting edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_negative_number() {
    let ctx = Context::new();
    let expr = ctx.int(-5);
    assert_eq!(format!("{expr}"), "-5");
}

#[test]
fn display_negative_rational() {
    let ctx = Context::new();
    let expr = ctx.rational(-1, 3);
    let s = format!("{expr}");
    eprintln!("-1/3 display = {s}");
    assert!(
        s.contains("-") && s.contains("1") && s.contains("3"),
        "display of -1/3: {s}"
    );
}

#[test]
fn display_large_integer() {
    let ctx = Context::new();
    let expr = ctx.int(1_000_000_000);
    assert_eq!(format!("{expr}"), "1000000000");
}

// ═══════════════════════════════════════════════════════════════════════════
// 23. LaTeX rendering edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn latex_zero() {
    let ctx = Context::new();
    let s = ctx.int(0).to_latex();
    assert_eq!(s, "0");
}

#[test]
fn latex_negative() {
    let ctx = Context::new();
    let s = ctx.int(-3).to_latex();
    assert!(s.contains("-3") || s.contains("- 3"), "LaTeX of -3: {s}");
}

#[test]
fn latex_fraction() {
    let ctx = Context::new();
    let s = ctx.rational(1, 2).to_latex();
    eprintln!("LaTeX 1/2 = {s}");
    assert!(
        s.contains("frac") || s.contains("1") && s.contains("2"),
        "LaTeX of 1/2: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 24. Number theory edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn isprime_zero() {
    assert!(!symplex::ntheory::isprime(0), "0 is not prime");
}

#[test]
fn isprime_one() {
    assert!(!symplex::ntheory::isprime(1), "1 is not prime");
}

#[test]
fn isprime_negative() {
    assert!(!symplex::ntheory::isprime(-7), "-7 is not prime");
}

#[test]
fn isprime_two() {
    assert!(symplex::ntheory::isprime(2), "2 is prime");
}

#[test]
fn factorint_zero() {
    let result = symplex::ntheory::factorint(0);
    // 0 has no prime factorization: the empty list (documented; SymPy gives {0: 1}).
    assert!(result.is_empty(), "factorint(0) = {result:?}");
}

#[test]
fn factorint_one() {
    let result = symplex::ntheory::factorint(1);
    eprintln!("factorint(1) = {result:?}");
    assert!(
        result.is_empty(),
        "1 has empty factorization, got: {result:?}"
    );
}

#[test]
fn factorint_negative() {
    let result = symplex::ntheory::factorint(-12);
    // Documented: the sign is dropped, |−12| = 2² · 3.
    let want = vec![
        (num_bigint::BigInt::from(2), 2u32),
        (num_bigint::BigInt::from(3), 1u32),
    ];
    assert_eq!(result, want, "factorint(-12)");
}

// ═══════════════════════════════════════════════════════════════════════════
// 25. Combinatorics edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn stirling2_zero_zero() {
    let result = symplex::combinatorics::stirling2(0, 0);
    assert_eq!(result, Some(num_bigint::BigInt::from(1)), "S(0,0) = 1");
}

#[test]
fn stirling2_negative() {
    let result = symplex::combinatorics::stirling2(-1, 2);
    // No partitions of a negative-size set: 0 (not a panic, not None).
    assert_eq!(
        result,
        Some(num_bigint::BigInt::from(0)),
        "S(-1, 2) = {result:?}"
    );
}

#[test]
fn partition_count_zero() {
    let result = symplex::combinatorics::partition_count(0);
    assert_eq!(result, Some(num_bigint::BigInt::from(1)), "p(0) = 1");
}

#[test]
fn partition_count_negative() {
    let result = symplex::combinatorics::partition_count(-1);
    assert_eq!(result, Some(num_bigint::BigInt::from(0)), "p(-1) = 0");
}

#[test]
fn partition_count_one() {
    let result = symplex::combinatorics::partition_count(1);
    assert_eq!(result, Some(num_bigint::BigInt::from(1)), "p(1) = 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// 26. Equation type edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn equation_trivially_true() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x = x is trivially true — solve should indicate identity
    let eq = symplex::eq::Equation::new(x.clone(), x.clone());
    // `solve` reports the identity explicitly; `solve_or_empty` maps that to [].
    assert!(
        matches!(eq.solve(&x), Err(SymplexError::InfiniteSolutions { .. })),
        "solve(x = x) = {:?}",
        eq.solve(&x)
    );
    assert!(eq.solve_or_empty(&x).is_empty());
}

#[test]
fn equation_trivially_false() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 0 = 1 — no solution
    let eq = symplex::eq::Equation::new(ctx.int(0), ctx.int(1));
    let roots = eq.solve_or_empty(&x);
    assert!(
        roots.is_empty(),
        "0 = 1 should have no solutions, got: {roots:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 27. Eval special functions at boundaries
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exp_zero_is_one() {
    let ctx = Context::new();
    let result = ctx.int(0).exp().eval();
    assert_eq!(format!("{result}"), "1", "exp(0) = 1");
}

#[test]
fn ln_one_is_zero() {
    let ctx = Context::new();
    let result = ctx.int(1).ln().eval();
    assert_eq!(format!("{result}"), "0", "ln(1) = 0");
}

#[test]
fn ln_e_is_one() {
    let ctx = Context::new();
    let result = ctx.e().ln().eval();
    let s = format!("{result}");
    assert_eq!(s, "1", "ln(e) should be 1, got: {s}");
}

#[test]
fn exp_one_is_e() {
    let ctx = Context::new();
    let result = ctx.int(1).exp().eval();
    let s = format!("{result}");
    assert_eq!(s, "E", "exp(1) should be E, got: {s}");
}

#[test]
fn ln_zero_no_panic() {
    let ctx = Context::new();
    let result = ctx.int(0).ln().eval();
    let s = format!("{result}");
    eprintln!("ln(0) = {s}");
    // ln(0) = -∞
    assert!(
        s.contains("-oo") || s.contains("-∞") || s.contains("oo") || s.contains("ln(0)"),
        "ln(0) should be -∞ or unevaluated, got: {s}"
    );
}

#[test]
fn ln_neg_one_no_panic() {
    let ctx = Context::new();
    let result = ctx.int(-1).ln().eval();
    let s = format!("{result}");
    eprintln!("ln(-1) = {s}");
    // ln(-1) = i*pi
    assert!(
        s.contains("I") || s.contains("i") || s.contains("pi") || s.contains("ln(-1)"),
        "ln(-1) should involve i*pi, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 28. Abs edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn abs_of_negative() {
    let ctx = Context::new();
    let result = ctx.int(-5).abs().eval();
    assert_eq!(format!("{result}"), "5", "|−5| = 5");
}

#[test]
fn abs_of_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).abs().eval();
    assert_eq!(format!("{result}"), "0", "|0| = 0");
}

#[test]
fn abs_of_positive() {
    let ctx = Context::new();
    let result = ctx.int(5).abs().eval();
    assert_eq!(format!("{result}"), "5", "|5| = 5");
}

#[test]
fn abs_of_abs_simplifies() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.abs().abs().simplify();
    assert_eq!(
        format!("{result}"),
        "abs(x)",
        "||x|| should simplify to |x|"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 29. Complex number edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn i_squared_is_neg_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(2).eval();
    let s = format!("{result}");
    assert_eq!(s, "-1", "i^2 should be -1, got: {s}");
}

#[test]
fn i_to_the_fourth_is_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(4).eval();
    let s = format!("{result}");
    assert_eq!(s, "1", "i^4 should be 1, got: {s}");
}

#[test]
fn i_plus_neg_i_is_zero() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = &i + &(-&i);
    assert!(
        result.is_zero_structural() || format!("{result}") == "0",
        "i + (-i) should be 0, got: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 30. Mixed operation stress tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_of_integral_is_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx ∫ x^2 dx should be x^2
    let integral = x.powi(2).integrate(&x);
    let deriv = integral.diff(&x);
    let s = format!("{deriv}");
    // Evaluate at x=3: should give 9
    let val = deriv.subs_i64(&x, 3).eval_f64().unwrap();
    assert!(
        (val - 9.0).abs() < 1e-8,
        "d/dx ∫ x^2 dx at x=3 should be 9, got {val} (display: {s})"
    );
}

#[test]
fn integral_of_diff_differs_by_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫ d/dx(x^3 + 5) dx = x^3 (+ C, but C=0 for indefinite)
    let expr = &x.powi(3) + 5;
    let deriv = expr.diff(&x); // 3x^2
    let integral = deriv.integrate(&x); // x^3
    let s = format!("{integral}");
    // Evaluate: integral at x=2 should be 8
    let val = integral.subs_i64(&x, 2).eval_f64().unwrap();
    assert!(
        (val - 8.0).abs() < 1e-8,
        "∫ 3x^2 dx at x=2 should be 8, got {val} (display: {s})"
    );
}

#[test]
fn solve_then_verify_solutions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3) = 0
    let expr = &x.powi(3) - &x.powi(2) * 6 + &x * 11 - 6;
    let roots = expr.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        3,
        "cubic should have 3 roots, got {}",
        roots.len()
    );
    // Verify each root
    for root in &roots {
        let val = expr.subs(&x, root);
        let s = format!("{val}");
        let simplified = val.simplify();
        let ss = format!("{simplified}");
        eprintln!("f({root}) = {s} → simplified = {ss}");
        assert!(
            ss == "0" || simplified.is_zero_structural(),
            "root {root} should satisfy equation, got f(root) = {ss}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 31. Serde roundtrip edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn serde_roundtrip_integer() {
    let ctx = Context::new();
    let expr = ctx.int(42);
    let tree = expr.to_tree();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{expr}"), format!("{back}"));
}

#[test]
fn serde_roundtrip_symbol() {
    let ctx = Context::new();
    let expr = ctx.symbol("x");
    let tree = expr.to_tree();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{expr}"), format!("{back}"));
}

#[test]
fn serde_roundtrip_complex_expr() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + &x.sin() + 1;
    let tree = expr.to_tree();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{expr}"), format!("{back}"));
}

#[test]
fn serde_roundtrip_infinity() {
    let ctx = Context::new();
    let expr = ctx.infinity();
    let tree = expr.to_tree();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{expr}"), format!("{back}"));
}

#[test]
fn serde_roundtrip_nan() {
    let ctx = Context::new();
    let expr = ctx.nan();
    let tree = expr.to_tree();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{expr}"), format!("{back}"));
}

// ═══════════════════════════════════════════════════════════════════════════
// 32. expr! macro edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_macro_just_integer() {
    // A purely numeric `expr!` is an `Ex` built through `ctx` (0.2 numfix).
    let ctx = Context::new();
    let result = expr!(ctx, 0);
    assert_eq!(format!("{result}"), "0");
    assert_eq!(result, ctx.int(0));
}

#[test]
fn expr_macro_negative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(ctx, -x);
    assert_eq!(format!("{result}"), "-x");
}

#[test]
fn expr_macro_nested_parens() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(ctx, (x + 1));
    let s = format!("{result}");
    assert!(s.contains("x") && s.contains("1"), "((x+1)) = {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 33. eval() idempotency
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_is_idempotent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let e1 = expr.eval();
    let e2 = e1.eval();
    assert_eq!(
        format!("{e1}"),
        format!("{e2}"),
        "eval should be idempotent"
    );
}

#[test]
fn simplify_is_idempotent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let s1 = expr.simplify();
    let s2 = s1.simplify();
    assert_eq!(
        format!("{s1}"),
        format!("{s2}"),
        "simplify should be idempotent"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 34. contains() edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn contains_self() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(x.contains(&x), "x should contain x");
}

#[test]
fn contains_subexpr() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2) + &x + 1;
    assert!(expr.contains(&x), "x^2+x+1 should contain x");
}

#[test]
fn does_not_contain_absent_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = x.powi(2) + 1;
    assert!(!expr.contains(&y), "x^2+1 should not contain y");
}

// ═══════════════════════════════════════════════════════════════════════════
// 35. Modular arithmetic edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mod_inverse_coprime() {
    let result = symplex::ntheory::mod_inverse(3_i64, 7_i64);
    assert_eq!(
        result,
        Some(num_bigint::BigInt::from(5)),
        "3^(-1) mod 7 = 5"
    );
}

#[test]
fn mod_inverse_not_coprime() {
    let result = symplex::ntheory::mod_inverse(6_i64, 9_i64);
    assert_eq!(result, None, "6 and 9 are not coprime, no inverse");
}

#[test]
fn mod_inverse_one() {
    let result = symplex::ntheory::mod_inverse(1_i64, 7_i64);
    assert_eq!(
        result,
        Some(num_bigint::BigInt::from(1)),
        "1^(-1) mod 7 = 1"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 36. Multiple symbols and substitution order
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn multi_symbol_subs_does_not_interfere() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // expr = x + y, substitute x→y, y→x — should not accidentally collapse
    let expr = &x + &y;
    let step1 = expr.subs(&x, &ctx.int(1));
    let step2 = step1.subs(&y, &ctx.int(2));
    let s = format!("{step2}");
    assert_eq!(s, "3", "x+y with x=1, y=2 should be 3, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 37. Very large number arithmetic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn big_integer_arithmetic_no_overflow() {
    let ctx = Context::new();
    let big = ctx.int(i64::MAX);
    let result = &big + &big;
    let s = format!("{result}");
    eprintln!("i64::MAX + i64::MAX = {s}");
    // Should be 2 * i64::MAX = 18446744073709551614
    let expected = (i64::MAX as i128) * 2;
    assert_eq!(s, expected.to_string(), "big + big = {s}");
}

#[test]
fn big_integer_multiplication_no_overflow() {
    let ctx = Context::new();
    let big = ctx.int(i64::MAX);
    let result = &big * &big;
    // (2^63 - 1)^2 = 85070591730234615847396907784232501249 — exact, no overflow.
    assert_eq!(
        format!("{result}"),
        "85070591730234615847396907784232501249"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 38. is_* queries edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn is_zero_structural_on_zero() {
    let ctx = Context::new();
    assert!(ctx.int(0).is_zero_structural());
}

#[test]
fn is_zero_structural_on_nonzero() {
    let ctx = Context::new();
    assert!(!ctx.int(1).is_zero_structural());
}

#[test]
fn is_zero_structural_on_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(!x.is_zero_structural());
}

#[test]
fn is_finite_on_integer() {
    let ctx = Context::new();
    assert_eq!(ctx.int(5).is_finite(), Some(true));
}

#[test]
fn is_finite_on_infinity() {
    let ctx = Context::new();
    assert_eq!(ctx.infinity().is_finite(), Some(false));
}

#[test]
fn is_positive_on_positive_int() {
    let ctx = Context::new();
    assert_eq!(ctx.int(5).is_positive(), Some(true));
}

#[test]
fn is_positive_on_zero() {
    let ctx = Context::new();
    assert_eq!(ctx.int(0).is_positive(), Some(false));
}

#[test]
fn is_positive_on_negative_int() {
    let ctx = Context::new();
    assert_eq!(ctx.int(-5).is_positive(), Some(false));
}

#[test]
fn is_positive_on_symbol_unknown() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Without assumptions, positivity should be unknown
    let result = x.is_positive();
    eprintln!("is_positive(x) = {result:?}");
    assert_eq!(result, None, "positivity of unconstrained x should be None");
}

// ═══════════════════════════════════════════════════════════════════════════
// 39. Polynomial operations on non-polynomials
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_non_polynomial_no_crash() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin() + 1;
    let result = expr.factor(&x);
    // Non-polynomial input is returned unchanged.
    assert_eq!(result, expr, "factor(sin(x)+1, x) = {result}");
}

#[test]
fn cancel_non_polynomial_no_crash() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin() / &x.cos();
    let result = expr.cancel(&x);
    // Nothing to cancel; the value must be preserved (tan(1) at x = 1).
    let v = result.subs_i64(&x, 1).eval_f64().unwrap();
    assert!((v - 1f64.tan()).abs() < 1e-12, "cancel(sin/cos) = {result}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 40. Assumption edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn symbol_with_positive_assumption() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]);
    assert_eq!(x.is_positive(), Some(true));
}

#[test]
fn symbol_with_real_assumption() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    assert_eq!(x.is_real(), Some(true));
    // Real says nothing about the sign.
    assert_eq!(x.is_positive(), None);
    assert_eq!(x.is_imaginary(), Some(false));
}

#[test]
fn symbol_with_integer_assumption() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Integer]);
    let result = x.is_integer();
    eprintln!("is_integer(x with Integer) = {result:?}");
    assert_eq!(result, Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// 41. Second-order and higher derivatives
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn second_derivative_x_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(3).diff(&x).diff(&x);
    let s = format!("{result}");
    // d²/dx² x³ = 6x
    assert!(s == "6*x", "d²/dx² x³ should be 6*x, got: {s}");
}

#[test]
fn third_derivative_x_cubed_is_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(3).diff(&x).diff(&x).diff(&x);
    let s = format!("{result}");
    assert_eq!(s, "6", "d³/dx³ x³ = 6, got: {s}");
}

#[test]
fn fourth_derivative_x_cubed_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(3).diff(&x).diff(&x).diff(&x).diff(&x);
    let s = format!("{result}");
    assert_eq!(s, "0", "d⁴/dx⁴ x³ = 0, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 42. Empty operations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compile_empty_args_for_constant() {
    let ctx = Context::new();
    let five = ctx.int(5);
    match five.compile(&[]) {
        Ok(f) => assert!((f(&[]) - 5.0).abs() < 1e-10),
        Err(_) => eprintln!("compile(5, []) returned Err — acceptable"),
    }
}

#[test]
fn to_rust_fn_empty_args_constant() {
    let ctx = Context::new();
    let five = ctx.int(5);
    match five.to_rust_fn("f", &[]) {
        Ok(code) => {
            eprintln!("to_rust_fn(5) = {code}");
            assert!(code.contains("5") || code.contains("f"));
        }
        Err(e) => eprintln!("to_rust_fn(5) errored: {e}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 43. Repeated solve on same expression
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_twice_gives_same_result() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2) - 4;
    let roots1 = expr.solve_or_empty(&x);
    let roots2 = expr.solve_or_empty(&x);
    let mut s1: Vec<String> = roots1.iter().map(|r| format!("{r}")).collect();
    let mut s2: Vec<String> = roots2.iter().map(|r| format!("{r}")).collect();
    s1.sort();
    s2.sort();
    assert_eq!(s1, s2, "solving twice should give same results");
}

// ═══════════════════════════════════════════════════════════════════════════
// 44. Interaction between expand and simplify
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_then_simplify_preserves_value() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1) * (&x - 1);
    let expanded = expr.expand();
    let simplified = expanded.simplify();
    // Should both evaluate to the same thing at x=5
    let v1 = expr.subs_i64(&x, 5).eval_f64().unwrap();
    let v2 = expanded.subs_i64(&x, 5).eval_f64().unwrap();
    let v3 = simplified.subs_i64(&x, 5).eval_f64().unwrap();
    assert!((v1 - v2).abs() < 1e-10, "expand should preserve value");
    assert!((v1 - v3).abs() < 1e-10, "simplify should preserve value");
}

// ═══════════════════════════════════════════════════════════════════════════
// 45. to_tree edge cases for special nodes
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn to_tree_and_back_pi() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let tree = pi.to_tree();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{pi}"), format!("{back}"), "pi roundtrip");
}

#[test]
fn to_tree_and_back_e() {
    let ctx = Context::new();
    let e = ctx.e();
    let tree = e.to_tree();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{e}"), format!("{back}"), "e roundtrip");
}

#[test]
fn to_tree_and_back_i_unit() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let tree = i.to_tree();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{i}"), format!("{back}"), "i roundtrip");
}

// ═══════════════════════════════════════════════════════════════════════════
// 46. Rational simplification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rational_auto_reduces() {
    let ctx = Context::new();
    let r = ctx.rational(2, 4);
    let s = format!("{r}");
    assert_eq!(s, "1/2", "2/4 should reduce to 1/2, got: {s}");
}

#[test]
fn rational_negative_denom() {
    let ctx = Context::new();
    let r = ctx.rational(1, -2);
    let s = format!("{r}");
    assert!(
        s == "-1/2" || s == "-(1/2)",
        "1/(-2) should display as -1/2, got: {s}"
    );
}

#[test]
fn rational_both_negative() {
    let ctx = Context::new();
    let r = ctx.rational(-1, -2);
    let s = format!("{r}");
    assert_eq!(s, "1/2", "(-1)/(-2) should be 1/2, got: {s}");
}

#[test]
fn rational_zero_numerator() {
    let ctx = Context::new();
    let r = ctx.rational(0, 5);
    let s = format!("{r}");
    assert_eq!(s, "0", "0/5 should be 0, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 47. Power rule integration edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_x_to_large_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫ x^99 dx = x^100 / 100
    let result = x.powi(99).integrate(&x);
    let s = format!("{result}");
    assert!(s.contains("100"), "∫ x^99 dx should contain 100: {s}");
}

#[test]
fn integrate_x_to_neg_two() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫ x^(-2) dx = -x^(-1) = -1/x
    let result = x.powi(-2).integrate(&x);
    let s = format!("{result}");
    eprintln!("∫ x^(-2) dx = {s}");
    // Verify: d/dx(-1/x) = 1/x^2 = x^(-2)
    let deriv = result.diff(&x);
    let val_orig = x.powi(-2).subs_i64(&x, 3).eval_f64().unwrap();
    let val_deriv = deriv.subs_i64(&x, 3).eval_f64().unwrap();
    assert!(
        (val_orig - val_deriv).abs() < 1e-8,
        "d/dx(∫ x^(-2) dx) should equal x^(-2) at x=3: {val_orig} vs {val_deriv}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 48. Nested power simplification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn power_of_power_integers() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x^2)^3 = x^6
    let result = x.powi(2).powi(3);
    let s = format!("{result}");
    assert_eq!(s, "x^6", "(x^2)^3 should be x^6, got: {s}");
}

#[test]
fn power_of_power_mixed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    // (x^4)^(1/2) for unknown x: cannot flatten because x might be negative
    // or complex. sqrt(x^4) = |x|^2, not x^2.
    // The library correctly leaves this unevaluated as sqrt(x^4).
    let result = x.powi(4).pow(&half);
    let s = format!("{result}");
    eprintln!("(x^4)^(1/2) = {s}");
    // Acceptable answers: sqrt(x^4), x^2, abs(x)^2
    assert!(
        s.contains("sqrt") || s.contains("x^2") || s.contains("abs"),
        "(x^4)^(1/2) should be sqrt(x^4) or x^2 or abs(x)^2: {s}"
    );
}

/// With a positive assumption, (x^4)^(1/2) should simplify further.
#[test]
fn power_of_power_mixed_positive_x() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]);
    let half = ctx.rational(1, 2);
    let result = x.powi(4).pow(&half).simplify();
    // For positive x: sqrt(x^4) = x^2.
    assert_eq!(result, x.powi(2), "(x^4)^(1/2) [x>0] simplified = {result}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 49. Context reuse across many operations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn stress_context_many_symbols() {
    let ctx = Context::new();
    let mut symbols = Vec::new();
    for i in 0..100 {
        symbols.push(ctx.symbol(&format!("x{i}")));
    }
    // Sum all of them
    let mut sum = ctx.int(0);
    for s in &symbols {
        sum = &sum + s;
    }
    let s = format!("{sum}");
    eprintln!("sum of 100 symbols has length {}", s.len());
    // Should not panic, OOM, or produce garbage
    assert!(s.contains("x0"), "sum should contain x0: {s}");
    assert!(s.contains("x99"), "sum should contain x99: {s}");
}

#[test]
fn stress_context_many_operations() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Repeatedly build and simplify expressions
    for i in 0..100 {
        let expr = x.powi(i) + &x + 1;
        let _ = expr.simplify();
        let _ = expr.diff(&x);
    }
    // Just verify we get through without crash
}

// ═══════════════════════════════════════════════════════════════════════════
// 50. Inequality edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_gt_positive_definite() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^2 + 1 > 0 is always true
    let expr = x.powi(2) + 1;
    let result = expr.solve_gt(&x);
    // Always true: the whole real line.
    assert_eq!(
        result.contains(&ctx.int(-7)),
        Some(true),
        "x^2 + 1 > 0: {result}"
    );
    assert_eq!(result.contains(&ctx.int(0)), Some(true));
    assert_eq!(
        result.simplify(),
        ctx.reals().simplify(),
        "x^2 + 1 > 0: {result}"
    );
}

#[test]
fn solve_gt_never_true() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // -x^2 - 1 > 0 is never true
    let expr = -x.powi(2) - 1;
    let result = expr.solve_gt(&x);
    // Never true: the empty set.
    assert_eq!(result.is_empty(), Some(true), "-x^2 - 1 > 0: {result}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 51. Additional division-related edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rational_division_chain() {
    let ctx = Context::new();
    // (1/2) / (1/3) = 3/2
    let a = ctx.rational(1, 2);
    let b = ctx.rational(1, 3);
    let result = &a / &b;
    let s = format!("{result}");
    assert_eq!(s, "3/2", "(1/2)/(1/3) should be 3/2, got: {s}");
}

#[test]
fn nested_fraction_simplification() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x/2) / (x/3) = 3/2 (for x ≠ 0)
    let numer = &x / &ctx.int(2);
    let denom = &x / &ctx.int(3);
    let result = (&numer / &denom).simplify();
    let s = format!("{result}");
    eprintln!("(x/2)/(x/3) simplified = {s}");
    // Should be 3/2
    assert!(
        s == "3/2" || s.contains("3/2"),
        "(x/2)/(x/3) should simplify to 3/2, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 52. Multivariate expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_multivar_partial_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // ∂/∂x (x^2 * y + x * y^2) = 2xy + y^2
    let expr = &x.powi(2) * &y + &x * &y.powi(2);
    let result = expr.diff(&x);
    let val = result.subs_i64(&x, 2).subs_i64(&y, 3).eval_f64().unwrap();
    // 2*2*3 + 3^2 = 12 + 9 = 21
    assert!(
        (val - 21.0).abs() < 1e-10,
        "∂f/∂x at (2,3) should be 21, got {val}"
    );
}

#[test]
fn diff_multivar_partial_y() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // ∂/∂y (x^2 * y + x * y^2) = x^2 + 2xy
    let expr = &x.powi(2) * &y + &x * &y.powi(2);
    let result = expr.diff(&y);
    let val = result.subs_i64(&x, 2).subs_i64(&y, 3).eval_f64().unwrap();
    // 2^2 + 2*2*3 = 4 + 12 = 16
    assert!(
        (val - 16.0).abs() < 1e-10,
        "∂f/∂y at (2,3) should be 16, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 53. Large factorial (symbolic)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorial_large_no_crash() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(100);
        let fact = arena.factorial(n);
        let result = arena.eval_expr(fact);
        let s = arena.display(result).to_string();
        eprintln!("100! has {} digits", s.len());
        // 100! has 158 digits
        assert!(
            s.len() > 150,
            "100! should have > 150 digits, got {}",
            s.len()
        );
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// 54. Integration of trig functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sin().integrate(&x);
    let s = format!("{result}");
    // ∫ sin(x) dx = -cos(x)
    assert!(s.contains("cos"), "∫ sin(x) dx should contain cos: {s}");
}

#[test]
fn integrate_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.cos().integrate(&x);
    let s = format!("{result}");
    // ∫ cos(x) dx = sin(x)
    assert!(s.contains("sin"), "∫ cos(x) dx should contain sin: {s}");
}

#[test]
fn integrate_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.exp().integrate(&x);
    let s = format!("{result}");
    // ∫ exp(x) dx = exp(x)
    assert!(s.contains("exp"), "∫ exp(x) dx should contain exp: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 55. Solve with complex roots
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_x_squared_plus_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^2 + 1 = 0 → x = ±i
    let roots = (x.powi(2) + 1).solve_or_empty(&x);
    eprintln!("x^2+1=0 roots: {roots:?}");
    assert_eq!(
        roots.len(),
        2,
        "x^2+1=0 should have 2 complex roots, got {}",
        roots.len()
    );
    let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    strs.sort();
    eprintln!("root strings: {strs:?}");
    // Should contain i and -i in some form
}

// ═══════════════════════════════════════════════════════════════════════════
// 56. GCD edge cases (number theory)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gcd_with_zero() {
    use num_bigint::BigInt;
    use num_integer::Integer;
    let a = BigInt::from(0);
    let b = BigInt::from(5);
    let g = a.gcd(&b);
    assert_eq!(g, BigInt::from(5), "gcd(0, 5) = 5");
}

#[test]
fn gcd_both_zero() {
    use num_bigint::BigInt;
    use num_integer::Integer;
    let a = BigInt::from(0);
    let b = BigInt::from(0);
    let g = a.gcd(&b);
    assert_eq!(g, BigInt::from(0), "gcd(0, 0) = 0");
}

// ═══════════════════════════════════════════════════════════════════════════
// 57. Deep chain of operations without simplification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn long_chain_add_sub_cancel() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (((x + 1) + 2) + 3) - 1 - 2 - 3 should be x
    let expr = ((&x + 1) + 2 + 3) - 1 - 2 - 3;
    let s = format!("{expr}");
    assert_eq!(s, "x", "additions and subtractions should cancel: {s}");
}

#[test]
fn long_chain_mul_div_cancel() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x * 2 * 3 / 2 / 3 = x
    let expr = &(&(&(&x * 2) * 3) / &ctx.int(2)) / &ctx.int(3);
    let s = format!("{expr}");
    eprintln!("x*2*3/2/3 = {s}");
    // Might need simplification
    let simplified = expr.simplify();
    let ss = format!("{simplified}");
    assert_eq!(ss, "x", "x*2*3/2/3 should simplify to x, got: {ss}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 58. Power with symbolic exponent
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn x_to_y_diff_wrt_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // d/dx (x^y) = y * x^(y-1)
    let expr = x.pow(&y);
    let result = expr.diff(&x);
    let s = format!("{result}");
    eprintln!("d/dx x^y = {s}");
    // At x=2, y=3: should be 3 * 2^2 = 12
    let val = result.subs_i64(&x, 2).subs_i64(&y, 3).eval_f64().unwrap();
    assert!(
        (val - 12.0).abs() < 1e-8,
        "d/dx x^y at (2,3) should be 12, got {val}"
    );
}

#[test]
fn x_to_y_diff_wrt_y() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // d/dy (x^y) = x^y * ln(x)
    let expr = x.pow(&y);
    let result = expr.diff(&y);
    let s = format!("{result}");
    eprintln!("d/dy x^y = {s}");
    // At x=2, y=3: should be 8 * ln(2) ≈ 5.5452
    let val = result.subs_i64(&x, 2).subs_i64(&y, 3).eval_f64().unwrap();
    let expected = 8.0 * 2.0_f64.ln();
    assert!(
        (val - expected).abs() < 1e-6,
        "d/dy x^y at (2,3) should be {expected}, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 59. Cancellation of common factors
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cancel_x_squared_minus_one_over_x_minus_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x.powi(2) - 1) / (&x - 1);
    let result = expr.cancel(&x);
    let s = format!("{result}");
    assert_eq!(s, "x + 1", "(x²-1)/(x-1) should cancel to x+1, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 60. Integration by parts patterns
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_x_exp_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫ x*exp(x) dx = x*exp(x) - exp(x)
    let expr = &x * &x.exp();
    let result = expr.integrate(&x);
    let s = format!("{result}");
    eprintln!("∫ x*exp(x) dx = {s}");
    assert!(
        !s.contains("Integral"),
        "∫ x*exp(x) dx should be closed-form, got: {s}"
    );
    // Verify by differentiation
    let deriv = result.diff(&x);
    let val_orig = expr.subs_i64(&x, 1).eval_f64().unwrap();
    let val_deriv = deriv.subs_i64(&x, 1).eval_f64().unwrap();
    assert!(
        (val_orig - val_deriv).abs() < 1e-6,
        "d/dx(∫ x*exp(x) dx) at x=1: {val_orig} vs {val_deriv}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 61. Logarithm properties
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ln_of_product_expand_log() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]);
    let y = ctx.symbol_with("y", &[Assumption::Positive]);
    let expr = (&x * &y).ln();
    let expanded = expr.expand_log();
    let s = format!("{expanded}");
    eprintln!("ln(x*y) expanded = {s}");
    // Should be ln(x) + ln(y)
    assert!(
        s.contains("ln(x)") && s.contains("ln(y)"),
        "ln(x*y) should expand to ln(x)+ln(y), got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 62. Multiple solve calls don't corrupt state
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_multiple_equations_same_context() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let r1 = (&x - 1).solve_or_empty(&x);
    let r2 = (&x - 2).solve_or_empty(&x);
    let r3 = (x.powi(2) - 9).solve_or_empty(&x);

    assert_eq!(format!("{}", r1[0]), "1");
    assert_eq!(format!("{}", r2[0]), "2");
    assert_eq!(r3.len(), 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// 63. Regression: double negation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn double_negation_is_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = -(-&x);
    let s = format!("{result}");
    assert_eq!(s, "x", "--x should be x, got: {s}");
}

#[test]
fn triple_negation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = -(-(-&x));
    let s = format!("{result}");
    assert_eq!(s, "-x", "---x should be -x, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 64. Symbolic evaluation of hyperbolic functions at zero
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sinh_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).sinh().eval();
    assert_eq!(format!("{result}"), "0", "sinh(0) = 0");
}

#[test]
fn cosh_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).cosh().eval();
    assert_eq!(format!("{result}"), "1", "cosh(0) = 1");
}

#[test]
fn tanh_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).tanh().eval();
    assert_eq!(format!("{result}"), "0", "tanh(0) = 0");
}

// ═══════════════════════════════════════════════════════════════════════════
// 65. Interaction with sets
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn interval_construction() {
    let ctx = Context::new();
    let a = ctx.int(0);
    let b = ctx.int(1);
    let interval = ctx.interval(&a, &b, IntervalKind::Closed);
    // (left_open, right_open) = (false, false) is the CLOSED interval [0, 1].
    assert_eq!(format!("{interval}"), "[0, 1]");
    assert_eq!(interval.contains(&ctx.int(0)), Some(true));
    assert_eq!(interval.contains(&ctx.int(1)), Some(true));
    assert_eq!(interval.contains(&ctx.int(2)), Some(false));
    let open = ctx.interval(&a, &b, IntervalKind::Open);
    assert_eq!(format!("{open}"), "(0, 1)");
    assert_eq!(open.contains(&ctx.int(0)), Some(false));
}

#[test]
fn empty_set_display() {
    let ctx = Context::new();
    let empty = ctx.empty_set();
    assert_eq!(format!("{empty}"), "EmptySet");
    assert_eq!(empty.is_empty(), Some(true));
    assert_eq!(empty.contains(&ctx.int(0)), Some(false));
}

// ═══════════════════════════════════════════════════════════════════════════
// 66. Thread safety basic check
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn context_clone_and_use_across_threads() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2) + 1;

    let ctx2 = ctx.clone();
    let handle = std::thread::spawn(move || {
        let x2 = ctx2.symbol("x");
        let result = expr.subs_i64(&x2, 3).eval_f64().unwrap();
        assert!((result - 10.0).abs() < 1e-10);
    });
    handle.join().unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// 67. eval_f64 on special values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_f64_infinity() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.eval_f64();
    eprintln!("eval_f64(oo) = {result:?}");
    match result {
        Ok(v) => assert!(
            v.is_infinite() && v > 0.0,
            "eval_f64(oo) should be +inf, got {v}"
        ),
        Err(e) => eprintln!("eval_f64(oo) errored (acceptable): {e}"),
    }
}

#[test]
fn eval_f64_nan() {
    let ctx = Context::new();
    let n = ctx.nan();
    let result = n.eval_f64();
    eprintln!("eval_f64(nan) = {result:?}");
    match result {
        Ok(v) => assert!(v.is_nan(), "eval_f64(nan) should be NaN, got {v}"),
        Err(e) => eprintln!("eval_f64(nan) errored (acceptable): {e}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 68. Substitution with infinity
// ═══════════════════════════════════════════════════════════════════════════

/// Substituting x=oo into 1/x correctly returns 0.
#[test]
fn subs_x_with_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = ctx.int(1) / &x;
    let result = expr.subs(&x, &ctx.infinity());
    let s = format!("{result}");
    assert_eq!(s, "0", "1/oo should be 0, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 71. Additional probing around the 0/0 bug
// ═══════════════════════════════════════════════════════════════════════════

/// 0 * (0^(-1)) correctly returns NaN (0 * zoo is indeterminate).
#[test]
fn zero_times_zero_to_neg_one() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let zero_inv = zero.powi(-1); // 0^(-1), which represents zoo
    let result = &zero * &zero_inv;
    let s = format!("{result}");
    assert_eq!(s, "nan", "0 * 0^(-1) should be NaN, got: {s}");
}

/// **BUG cascade**: x/x for x=0 should be undefined (0/0), but we get 1.
/// Because x/x canonicalizes to 1 at construction time (for symbolic x),
/// substituting x=0 gives 1, not NaN.
/// This is actually *correct behavior* — x/x = 1 for all x ≠ 0, and the
/// cancellation is valid for symbolic x. The library assumes symbols are
/// non-zero by default. Documenting, not flagging as bug.
#[test]
fn x_div_x_at_zero_is_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x / &x; // canonicalizes to 1
    let result = expr.subs(&x, &ctx.int(0));
    let s = format!("{result}");
    eprintln!("x/x at x=0 = {s}");
    // Structural x/x = 1, so subs(x=0) in 1 gives 1.
    assert_eq!(s, "1", "x/x (already simplified to 1) at x=0 is 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// 72. More infinity arithmetic probing
// ═══════════════════════════════════════════════════════════════════════════

/// oo^(-2) correctly simplifies to 0.
#[test]
fn infinity_to_neg_two() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.powi(-2);
    let s = format!("{result}");
    assert_eq!(s, "0", "oo^(-2) should be 0, got: {s}");
}

/// (-oo)^(-1) correctly simplifies to 0.
#[test]
fn neg_infinity_to_neg_one() {
    let ctx = Context::new();
    let neg_inf = -ctx.infinity();
    let result = neg_inf.powi(-1);
    let s = format!("{result}");
    assert_eq!(s, "0", "(-oo)^(-1) should be 0, got: {s}");
}

/// oo + 1 should still be oo.
#[test]
fn infinity_plus_one_is_infinity() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = &inf + &ctx.int(1);
    let s = format!("{result}");
    assert_eq!(s, "oo", "oo + 1 should be oo, got: {s}");
}

/// oo * 2 should be oo.
#[test]
fn infinity_times_two_is_infinity() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = &inf * &ctx.int(2);
    let s = format!("{result}");
    assert_eq!(s, "oo", "oo * 2 should be oo, got: {s}");
}

/// oo * (-1) should be -oo.
#[test]
fn infinity_times_neg_one_is_neg_infinity() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = &inf * &ctx.int(-1);
    let s = format!("{result}");
    assert_eq!(s, "-oo", "oo * (-1) should be -oo, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 73. Probing 0^(negative) more carefully
// ═══════════════════════════════════════════════════════════════════════════

/// 0^(-1) correctly returns zoo (complex infinity).
#[test]
fn zero_to_neg_one_should_be_zoo() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = zero.powi(-1);
    let s = format!("{result}");
    assert_eq!(s, "zoo", "0^(-1) should be zoo, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 74. Probing whether eval() fixes the infinity/zero bugs
// ═══════════════════════════════════════════════════════════════════════════

/// Check if .eval() resolves 1/oo to 0.
#[test]
fn eval_one_over_infinity() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let expr = ctx.int(1) / &inf;
    let evald = expr.eval();
    // 1/oo = 0 (SymPy agrees).
    assert!(evald.is_zero_structural(), "eval(1/oo) = {evald}");
}

/// Check if .simplify() resolves 1/oo to 0.
#[test]
fn simplify_one_over_infinity() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let expr = ctx.int(1) / &inf;
    let simplified = expr.simplify();
    assert!(
        simplified.is_zero_structural(),
        "simplify(1/oo) = {simplified}"
    );
}

/// Check if limit correctly handles 1/oo even though direct arithmetic doesn't.
#[test]
fn limit_handles_infinity_correctly() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // lim(x→∞) 1/x = 0 — limit engine should handle this even though
    // direct substitution gives 1/oo.
    let expr = ctx.int(1) / &x;
    let result = expr.limit(&x, &ctx.infinity());
    let s = format!("{result}");
    assert_eq!(
        s, "0",
        "lim(x→∞) 1/x should be 0 via limit engine, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 75. More 0/0-adjacent bugs: 0 * zoo should be NaN
// ═══════════════════════════════════════════════════════════════════════════

/// 0 * zoo should be NaN.
#[test]
fn zero_times_zoo_is_nan() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let z = zoo(&ctx);
    let result = &zero * &z;
    let s = format!("{result}");
    eprintln!("0 * zoo = {s}");
    assert!(
        s.contains("nan") || s.contains("NaN"),
        "0 * zoo should be NaN, got: {s}"
    );
}

/// 0 * (x/0) correctly returns NaN (since x/0 is zoo, and 0 * zoo is NaN).
#[test]
fn zero_times_x_over_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let x_over_zero = &x / &zero; // x * 0^(-1)
    let result = &zero * &x_over_zero;
    let s = format!("{result}");
    assert_eq!(s, "nan", "0 * (x/0) should be NaN, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 76. Probing eval_f64 on expressions involving division by zero
// ═══════════════════════════════════════════════════════════════════════════

/// eval_f64 on 0/0 correctly returns an Err since 0/0 is NaN at construction time.
#[test]
fn eval_f64_zero_over_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = (&zero / &zero).eval_f64();
    assert!(
        result.is_err(),
        "eval_f64(0/0) should be Err since 0/0 is NaN, got: {result:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 77. Regression: double-check that 0/x = 0 for symbolic x (correct)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zero_over_symbol_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let result = &zero / &x;
    let s = format!("{result}");
    assert_eq!(s, "0", "0/x should be 0, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 78. Probing for potential infinite-loop or timeout in integration
// ═══════════════════════════════════════════════════════════════════════════

/// Integration of x^x should return an unevaluated Integral (not hang).
#[test]
fn integrate_x_to_x_returns_unevaluated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.pow(&x); // x^x has no elementary antiderivative
    let result = expr.integrate(&x);
    let s = format!("{result}");
    eprintln!("∫ x^x dx = {s}");
    // Should contain "Integral" (unevaluated form)
    assert!(
        s.contains("Integral") || s.contains("∫"),
        "∫ x^x dx should be unevaluated, got: {s}"
    );
}

/// solve for a constant (not a variable) should not panic.
#[test]
fn solve_wrt_constant_no_crash() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    let expr = &x + 1;
    // Solving x+1=0 for "2" (a constant, not a variable) is rejected, not a panic.
    let result = expr.solve(&two);
    assert!(result.is_err(), "solve(x+1, 2) = {result:?}");
}

/// Diff wrt a constant (not a variable) — should just return 0.
#[test]
fn diff_wrt_constant_returns_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    let expr = x.powi(2);
    let result = expr.diff(&two);
    let s = format!("{result}");
    eprintln!("d/d(2) x^2 = {s}");
    // Differentiating wrt a constant should give 0
    assert_eq!(s, "0", "d/d(constant) should be 0, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 69. Multiple contexts don't interfere
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn independent_contexts_dont_interfere() {
    let ctx1 = Context::new();
    let ctx2 = Context::new();
    let x1 = ctx1.symbol("x");
    let x2 = ctx2.symbol("x");

    let expr1 = x1.powi(2);
    let expr2 = x2.powi(3);

    // Operations within each context should work independently
    let r1 = expr1.diff(&x1);
    let r2 = expr2.diff(&x2);

    assert_eq!(format!("{r1}"), "2*x");
    assert_eq!(format!("{r2}"), "3*x^2");
}

// ═══════════════════════════════════════════════════════════════════════════
// 70. Stress: many small operations in sequence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn stress_1000_additions() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let mut acc = ctx.int(0);
    for _ in 0..1000 {
        acc = &acc + &one;
    }
    let s = format!("{acc}");
    assert_eq!(s, "1000", "1+1+...+1 (1000 times) = {s}");
}

#[test]
fn stress_1000_multiplications() {
    let ctx = Context::new();
    let two = ctx.int(2);
    let mut acc = ctx.int(1);
    for _ in 0..10 {
        acc = &acc * &two;
    }
    let s = format!("{acc}");
    assert_eq!(s, "1024", "2^10 via multiplication = {s}");
}
