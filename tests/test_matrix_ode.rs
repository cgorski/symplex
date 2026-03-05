//! Comprehensive tests for Matrix, ODE, and Equation types.

use symplex::matrix::{Matrix, jacobian};
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Matrix construction
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_zeros() {
    let m = Matrix::zeros(3, 3);
    assert_eq!(m.nrows(), 3);
    assert_eq!(m.ncols(), 3);
    assert_eq!(format!("{}", m.get(1, 1)), "0");
}

#[test]
fn matrix_identity_3x3() {
    let m = Matrix::identity(3);
    assert_eq!(format!("{}", m.get(0, 0)), "1");
    assert_eq!(format!("{}", m.get(0, 1)), "0");
    assert_eq!(format!("{}", m.get(2, 2)), "1");
}

#[test]
fn matrix_from_fn() {
    let m = Matrix::from_fn(2, 3, |i, j| symplex::int((i * 3 + j + 1) as i64));
    assert_eq!(format!("{}", m.get(0, 0)), "1");
    assert_eq!(format!("{}", m.get(0, 2)), "3");
    assert_eq!(format!("{}", m.get(1, 0)), "4");
    assert_eq!(format!("{}", m.get(1, 2)), "6");
}

#[test]
fn matrix_row_col_vector() {
    let r = Matrix::row_vector(vec![symplex::int(1), symplex::int(2), symplex::int(3)]);
    assert_eq!(r.nrows(), 1);
    assert_eq!(r.ncols(), 3);

    let c = Matrix::col_vector(vec![symplex::int(1), symplex::int(2)]);
    assert_eq!(c.nrows(), 2);
    assert_eq!(c.ncols(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Matrix operations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_transpose() {
    let m = Matrix::from_fn(2, 3, |i, j| symplex::int((i * 3 + j) as i64));
    let t = m.transpose();
    assert_eq!(t.nrows(), 3);
    assert_eq!(t.ncols(), 2);
    assert_eq!(format!("{}", t.get(0, 0)), format!("{}", m.get(0, 0)));
    assert_eq!(format!("{}", t.get(1, 0)), format!("{}", m.get(0, 1)));
}

#[test]
fn matrix_add() {
    let a = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(3), symplex::int(4)],
    ]);
    let b = Matrix::new(vec![
        vec![symplex::int(10), symplex::int(20)],
        vec![symplex::int(30), symplex::int(40)],
    ]);
    let c = a.add(&b);
    assert_eq!(format!("{}", c.get(0, 0)), "11");
    assert_eq!(format!("{}", c.get(1, 1)), "44");
}

#[test]
fn matrix_scale() {
    let m = Matrix::identity(2);
    let scaled = m.scale(&symplex::int(5));
    assert_eq!(format!("{}", scaled.get(0, 0)), "5");
    assert_eq!(format!("{}", scaled.get(0, 1)), "0");
}

#[test]
fn matrix_matmul_identity() {
    let id = Matrix::identity(3);
    let m = Matrix::from_fn(3, 3, |i, j| symplex::int((i * 3 + j + 1) as i64));
    let result = id.matmul(&m);
    assert_eq!(format!("{}", result.get(0, 0)), format!("{}", m.get(0, 0)));
    assert_eq!(format!("{}", result.get(2, 2)), format!("{}", m.get(2, 2)));
}

#[test]
fn matrix_matmul_2x2() {
    let a = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(3), symplex::int(4)],
    ]);
    let b = Matrix::new(vec![
        vec![symplex::int(5), symplex::int(6)],
        vec![symplex::int(7), symplex::int(8)],
    ]);
    let c = a.matmul(&b);
    // [1*5+2*7, 1*6+2*8] = [19, 22]
    // [3*5+4*7, 3*6+4*8] = [43, 50]
    assert_eq!(format!("{}", c.get(0, 0)), "19");
    assert_eq!(format!("{}", c.get(0, 1)), "22");
    assert_eq!(format!("{}", c.get(1, 0)), "43");
    assert_eq!(format!("{}", c.get(1, 1)), "50");
}

#[test]
fn matrix_det_2x2() {
    let m = Matrix::new(vec![
        vec![symplex::int(3), symplex::int(7)],
        vec![symplex::int(1), symplex::int(5)],
    ]);
    assert_eq!(format!("{}", m.det()), "8");
}

#[test]
fn matrix_det_3x3_singular() {
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2), symplex::int(3)],
        vec![symplex::int(4), symplex::int(5), symplex::int(6)],
        vec![symplex::int(7), symplex::int(8), symplex::int(9)],
    ]);
    assert_eq!(format!("{}", m.det()), "0");
}

#[test]
fn matrix_det_3x3_nonsingular() {
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2), symplex::int(3)],
        vec![symplex::int(0), symplex::int(1), symplex::int(4)],
        vec![symplex::int(5), symplex::int(6), symplex::int(0)],
    ]);
    let det = m.det();
    // det = 1(0-24) - 2(0-20) + 3(0-5) = -24 + 40 - 15 = 1
    assert_eq!(format!("{det}"), "1");
}

#[test]
fn matrix_trace() {
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(0)],
        vec![symplex::int(0), symplex::int(4)],
    ]);
    assert_eq!(format!("{}", m.trace()), "5");
}

// ═══════════════════════════════════════════════════════════════════════════
// Matrix calculus
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_diff() {
    let x = symplex::var("x");
    let m = Matrix::new(vec![vec![x.powi(2), x.sin()], vec![x.cos(), x.exp()]]);
    let dm = m.diff(&x);
    let s00 = format!("{}", dm.get(0, 0));
    assert_eq!(s00, "2*x", "d/dx(x²) should be exactly 2*x, got: {s00}");
    let s01 = format!("{}", dm.get(0, 1));
    assert!(s01.contains("cos"), "d/dx(sin(x))={s01}");
}

#[test]
fn matrix_subs() {
    let x = symplex::var("x");
    let m = Matrix::new(vec![vec![x.powi(2), &x + 1]]);
    let at2 = m.subs(&x, &symplex::int(2));
    assert_eq!(format!("{}", at2.get(0, 0)), "4");
    assert_eq!(format!("{}", at2.get(0, 1)), "3");
}

#[test]
fn jacobian_2x2() {
    let x = symplex::var("x");
    let y = symplex::var("y");
    let f1 = &x.powi(2) * &y;
    let f2 = &x + &y.powi(3);
    let j = jacobian(&[f1, f2], &[x.clone(), y.clone()]);
    assert_eq!(j.nrows(), 2);
    assert_eq!(j.ncols(), 2);
    // J[0,0] = d/dx(x²y) = 2xy
    let s = format!("{}", j.get(0, 0));
    assert!(s.contains("x") && s.contains("y"), "J[0,0]={s}");
    // J[1,1] = d/dy(y³) contains y
    let s11 = format!("{}", j.get(1, 1));
    assert!(s11.contains("y"), "J[1,1]={s11}");
}

#[test]
fn matrix_display() {
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(3), symplex::int(4)],
    ]);
    let s = format!("{m}");
    // Verify all four entries appear and the matrix renders with structure
    assert!(s.contains("1") && s.contains("2") && s.contains("3") && s.contains("4"),
        "display should contain all entries 1,2,3,4: {s}");
    assert!(s.contains('[') || s.contains('|') || s.contains('\n'),
        "display should have matrix structure (brackets, pipes, or newlines): {s}");
}

#[test]
fn matrix_map() {
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(4)],
        vec![symplex::int(9), symplex::int(16)],
    ]);
    let sqrt_m = m.map(|e| e.sqrt().eval());
    assert_eq!(format!("{}", sqrt_m.get(0, 0)), "1");
    assert_eq!(format!("{}", sqrt_m.get(0, 1)), "2");
    assert_eq!(format!("{}", sqrt_m.get(1, 0)), "3");
    assert_eq!(format!("{}", sqrt_m.get(1, 1)), "4");
}

// ═══════════════════════════════════════════════════════════════════════════
// matrix! macro
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_macro_symbolic() {
    let x = symplex::var("x");
    let m = matrix![[x, 0], [0, x ^ 2]];
    let det = m.det();
    let s = format!("{det}");
    assert!(s.contains("x"), "det should involve x: {s}");
}

#[test]
fn matrix_macro_constants() {
    let m = matrix![[pi, 0], [0, E]];
    assert_eq!(format!("{}", m.get(0, 0)), "pi");
    assert_eq!(format!("{}", m.get(1, 1)), "E");
}

// ═══════════════════════════════════════════════════════════════════════════
// Equation type
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn equation_display() {
    let x = symplex::var("x");
    let eq = Equation::new(&x + 1, symplex::int(5));
    let s = format!("{eq}");
    assert!(s.contains("="), "should display equation: {s}");
}

#[test]
fn equation_solve() {
    let x = symplex::var("x");
    let eq = Equation::new(x.powi(2), symplex::int(4));
    let roots = eq.solve_or_empty(&x);
    assert_eq!(roots.len(), 2);
}

#[test]
fn equation_subs_check() {
    let x = symplex::var("x");
    let eq = Equation::new(&x * 3, symplex::int(12));
    let at4 = eq.subs_i64(&x, 4);
    assert_eq!(at4.is_satisfied(), Some(true));
}

#[test]
fn equation_simplify() {
    let x = symplex::var("x");
    let eq = Equation::new(&x.sin().powi(2) + &x.cos().powi(2), symplex::int(1));
    let simplified = eq.simplify();
    assert_eq!(format!("{}", simplified.lhs), "1");
}

#[test]
fn equation_to_expr() {
    let x = symplex::var("x");
    let eq = Equation::new(&x + 5, symplex::int(10));
    let expr = eq.to_expr();
    let roots = expr.solve_or_empty(&x);
    assert_eq!(format!("{}", roots[0]), "5");
}

// ═══════════════════════════════════════════════════════════════════════════
// eq! macro
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eq_macro_linear() {
    let x = symplex::var("x");
    let equation = eq!(x + 1 = 5);
    let roots = equation.solve_or_empty(&x);
    assert_eq!(roots.len(), 1);
    assert_eq!(format!("{}", roots[0]), "4");
}

#[test]
fn eq_macro_quadratic() {
    let x = symplex::var("x");
    let equation = eq!(x ^ 2 = 9);
    let roots = equation.solve_or_empty(&x);
    assert_eq!(roots.len(), 2);
}

#[test]
fn eq_macro_with_rational() {
    let x = symplex::var("x");
    let equation = eq!(2 * x = 1);
    let roots = equation.solve_or_empty(&x);
    assert_eq!(roots.len(), 1);
    assert_eq!(format!("{}", roots[0]), "1/2");
}

// ═══════════════════════════════════════════════════════════════════════════
// ODE (arena-level, since dsolve requires Derivative nodes)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_type_exists() {
    // Verify ODE module is publicly accessible
    let _ = std::mem::size_of::<symplex::ode::OdeResult>();
}

#[test]
fn ode_via_arena_simple() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let x = arena.symbol("x");
        let y = arena.symbol("y");
        // Build y' (Derivative node)
        let dy = arena.formal_diff(y, x);
        // y' - x = 0 → y = x²/2 + C1
        let eq = arena.sub(dy, x);
        let result = symplex::ode::dsolve(arena, eq, y, x)
            .expect("dsolve should handle y' - x = 0 via arena");
        let s = arena.display(result.solution).to_string();
        assert!(s.contains("C1"), "should have constant: {s}");
        assert!(s.contains("x"), "should contain x: {s}");
    });
}

#[test]
fn ode_via_arena_exponential() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let x = arena.symbol("x");
        let y = arena.symbol("y");
        let dy = arena.formal_diff(y, x);
        let two = arena.int(2);
        let two_y = arena.mul(&[two, y]);
        // y' + 2y = 0 → y = C1*e^(-2x)
        let eq = arena.add(&[dy, two_y]);
        let result = symplex::ode::dsolve(arena, eq, y, x)
            .expect("dsolve should handle y' + 2y = 0 via arena");
        let s = arena.display(result.solution).to_string();
        assert!(s.contains("exp"), "should contain exp: {s}");
        assert!(s.contains("C1"), "should have constant: {s}");
    });
}
