//! Tests for macro improvements (expr! constants, rationals, matrix!, eq!).

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// expr! with constants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_pi() {
    let __ctx = Context::new();
    let result = expr!(pi);
    assert_eq!(format!("{result}"), "pi");
}

#[test]
fn expr_e_constant() {
    let __ctx = Context::new();
    let result = expr!(E);
    assert_eq!(format!("{result}"), "E");
}

#[test]
fn expr_imaginary_unit() {
    let __ctx = Context::new();
    let result = expr!(I);
    assert_eq!(format!("{result}"), "I");
}

#[test]
fn expr_sin_pi() {
    let __ctx = Context::new();
    let result = expr!(sin(pi)).eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn expr_cos_pi() {
    let __ctx = Context::new();
    let result = expr!(cos(pi)).eval();
    assert_eq!(format!("{result}"), "-1");
}

#[test]
fn expr_exp_i_pi() {
    let __ctx = Context::new();
    let result = expr!(exp(I * pi)).eval();
    assert_eq!(format!("{result}"), "-1");
}

#[test]
fn expr_euler_identity() {
    let __ctx = Context::new();
    let result = expr!(exp(I * pi) + 1).eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn expr_i_squared() {
    let __ctx = Context::new();
    let result = expr!(I ^ 2);
    assert_eq!(format!("{result}"), "-1");
}

#[test]
fn expr_pi_in_expression() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let result = expr!(x + pi);
    let s = format!("{result}");
    assert!(s.contains("pi") && s.contains("x"), "got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// expr! with rationals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_one_half() {
    let __ctx = Context::new();
    let result = expr!(1 / 2);
    assert_eq!(format!("{result}"), "1/2");
}

#[test]
fn expr_three_quarters() {
    let __ctx = Context::new();
    let result = expr!(3 / 4);
    assert_eq!(format!("{result}"), "3/4");
}

#[test]
fn expr_rational_reduces() {
    let __ctx = Context::new();
    let result = expr!(6 / 4);
    assert_eq!(format!("{result}"), "3/2");
}

#[test]
fn expr_rational_in_expression() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let result = expr!(1 / 2 * x);
    let s = format!("{result}");
    assert!(s.contains("1/2") && s.contains("x"), "got: {s}");
}

#[test]
fn expr_rational_addition() {
    let __ctx = Context::new();
    let result = expr!(1 / 2 + 1 / 3);
    assert_eq!(format!("{result}"), "5/6");
}

#[test]
fn expr_x_to_half_power() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let result = expr!(x ^ (1 / 2));
    let s = format!("{result}");
    // x^(1/2) should display as sqrt(x) or x^(1/2)
    assert!(s.contains("x"), "got: {s}");
}

#[test]
fn expr_negative_rational() {
    let __ctx = Context::new();
    // Note: -1/2 parses as (-1)/2 due to precedence (unary minus binds tighter than /).
    // Use explicit parentheses -(1/2) to get the rational -1/2.
    let result = expr!(-(1 / 2));
    assert_eq!(format!("{result}"), "-1/2");
}

// ═══════════════════════════════════════════════════════════════════════════
// expr! with log(x, base)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_log_base_2() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let result = expr!(log(x, 2));
    let s = format!("{result}");
    assert!(s.contains("ln"), "log(x,2) should use ln: {s}");
}

#[test]
fn expr_log_base_10() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let result = expr!(log(x, 10));
    let s = format!("{result}");
    assert!(s.contains("ln"), "log(x,10) should use ln: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// eq! macro
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eq_basic() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let equation = eq!(x + 1 = 5);
    let s = format!("{equation}");
    assert!(s.contains("="), "should display as equation: {s}");
}

#[test]
fn eq_solve() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let equation = eq!(x + 1 = 5);
    let roots = equation.solve_or_empty(&x);
    assert_eq!(roots.len(), 1);
    assert_eq!(format!("{}", roots[0]), "4");
}

#[test]
fn eq_quadratic() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let equation = eq!(x ^ 2 = 9);
    let roots = equation.solve_or_empty(&x);
    assert_eq!(roots.len(), 2);
}

#[test]
fn eq_with_rationals() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let equation = eq!(x = 1 / 2);
    let roots = equation.solve_or_empty(&x);
    assert_eq!(roots.len(), 1);
    assert_eq!(format!("{}", roots[0]), "1/2");
}

#[test]
fn eq_with_pi() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let equation = eq!(sin(x) = 0);
    let s = format!("{equation}");
    assert!(s.contains("sin") && s.contains("="), "got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// matrix! macro
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_2x2_numeric() {
    let __ctx = Context::new();
    let m = matrix![[1, 2], [3, 4]];
    assert_eq!(m.nrows(), 2);
    assert_eq!(m.ncols(), 2);
    assert_eq!(format!("{}", m.get(0, 0)), "1");
    assert_eq!(format!("{}", m.get(1, 1)), "4");
}

#[test]
fn matrix_with_expressions() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let m = matrix![[x, sin(x)], [cos(x), x ^ 2]];
    assert_eq!(m.nrows(), 2);
    assert_eq!(m.ncols(), 2);
    let s = format!("{}", m.get(0, 1));
    assert!(s.contains("sin"), "got: {s}");
}

#[test]
fn matrix_with_constants() {
    let __ctx = Context::new();
    let m = matrix![[pi, E], [I, 0]];
    assert_eq!(format!("{}", m.get(0, 0)), "pi");
    assert_eq!(format!("{}", m.get(0, 1)), "E");
    assert_eq!(format!("{}", m.get(1, 0)), "I");
}

#[test]
fn matrix_with_rationals() {
    let __ctx = Context::new();
    let m = matrix![[1 / 2, 0], [0, 1 / 2]];
    assert_eq!(format!("{}", m.get(0, 0)), "1/2");
}

#[test]
fn matrix_det() {
    let __ctx = Context::new();
    let m = matrix![[3, 7], [1, 5]];
    let det = m.det().unwrap();
    assert_eq!(format!("{det}"), "8");
}

#[test]
fn matrix_1x1() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let m = matrix![[x ^ 2 + 1]];
    assert_eq!(m.nrows(), 1);
    assert_eq!(m.ncols(), 1);
}

#[test]
fn matrix_3x3_identity() {
    let __ctx = Context::new();
    let m = matrix![[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let det = m.det().unwrap();
    assert_eq!(format!("{det}"), "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Combined workflows
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_eq_with_constants() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let equation = eq!(x ^ 2 + 1 = 0);
    let roots = equation.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "x²+1=0 should have complex roots");
}

#[test]
fn workflow_matrix_jacobian() {
    let __ctx = Context::new();
    use symplex::matrix::jacobian;
    let x = __ctx.symbol("x");
    let y = __ctx.symbol("y");
    let f1 = expr!(x ^ 2 + y);
    let f2 = expr!(x * y);
    let j = jacobian(&[&f1, &f2], &[&x, &y]);
    assert_eq!(j.nrows(), 2);
    assert_eq!(j.ncols(), 2);
}

#[test]
fn workflow_rational_solve() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let equation = eq!(2 * x = 1);
    let roots = equation.solve_or_empty(&x);
    assert_eq!(roots.len(), 1);
    assert_eq!(format!("{}", roots[0]), "1/2");
}

#[test]
fn workflow_euler_in_matrix() {
    let __ctx = Context::new();
    let m = matrix![[exp(I * pi), 0], [0, 1]];
    let evald = m.eval();
    assert_eq!(format!("{}", evald.get(0, 0)), "-1");
}
