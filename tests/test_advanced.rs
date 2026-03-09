//! Integration tests for advanced features (Cycles 16-18).

// ═══════════════════════════════════════════════════════════════════════════
// Matrix
// ═══════════════════════════════════════════════════════════════════════════

use symplex::prelude::*;
#[test]
fn matrix_identity_times_vector() {
    let ctx = Context::new();
    use symplex::matrix::Matrix;
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let id = Matrix::identity(&ctx, 2);
    let v = Matrix::col_vector(vec![x.clone(), y.clone()]);
    let result = id.matmul(&v).unwrap();
    assert_eq!(format!("{}", result.get(0, 0)), "x");
    assert_eq!(format!("{}", result.get(1, 0)), "y");
}

#[test]
fn matrix_det_2x2_numeric() {
    let ctx = Context::new();
    use symplex::matrix::Matrix;
    let m = Matrix::new(vec![
        vec![ctx.int(3), ctx.int(7)],
        vec![ctx.int(1), ctx.int(5)],
    ]).unwrap();
    let det = m.det().unwrap();
    // 3*5 - 7*1 = 8
    assert_eq!(format!("{det}"), "8");
}

#[test]
fn matrix_jacobian() {
    let ctx = Context::new();
    use symplex::matrix::jacobian;
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f1 = &x.powi(2) * &y;
    let f2 = &x + &y.powi(3);
    let j = jacobian(&[&f1, &f2], &[&x, &y]);
    assert_eq!(j.nrows(), 2);
    assert_eq!(j.ncols(), 2);

    // ∂(x²y)/∂x = 2xy
    let j00 = format!("{}", j.get(0, 0));
    assert!(
        j00.contains("2") && j00.contains("x") && j00.contains("y"),
        "∂(x²y)/∂x should be 2*x*y, got: {j00}"
    );

    // ∂(x²y)/∂y = x²
    let j01 = format!("{}", j.get(0, 1));
    assert_eq!(j01, "x^2", "∂(x²y)/∂y should be x^2, got: {j01}");

    // ∂(x+y³)/∂x = 1
    let j10 = format!("{}", j.get(1, 0));
    assert_eq!(j10, "1", "∂(x+y³)/∂x should be 1, got: {j10}");

    // ∂(x+y³)/∂y = 3y²
    let j11 = format!("{}", j.get(1, 1));
    assert!(
        j11.contains("3") && j11.contains("y"),
        "∂(x+y³)/∂y should be 3*y^2, got: {j11}"
    );
}

#[test]
fn matrix_trace() {
    let ctx = Context::new();
    use symplex::matrix::Matrix;
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3), ctx.int(4)],
    ]).unwrap();
    let tr = m.trace().unwrap();
    assert_eq!(format!("{tr}"), "5");
}

#[test]
fn matrix_add_numeric() {
    let ctx = Context::new();
    use symplex::matrix::Matrix;
    let m1 = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3), ctx.int(4)],
    ]).unwrap();
    let m2 = Matrix::new(vec![
        vec![ctx.int(10), ctx.int(20)],
        vec![ctx.int(30), ctx.int(40)],
    ]).unwrap();
    let sum = m1.add(&m2).unwrap();
    assert_eq!(format!("{}", sum.get(0, 0)), "11");
    assert_eq!(format!("{}", sum.get(1, 1)), "44");
}

#[test]
fn matrix_diff() {
    let ctx = Context::new();
    use symplex::matrix::Matrix;
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![x.powi(2), x.sin()]]).unwrap();
    let dm = m.diff(&x);
    // d/dx(x²) = 2*x
    let s0 = format!("{}", dm.get(0, 0));
    assert_eq!(s0, "2*x", "d/dx(x²) should be exactly 2*x, got: {s0}");
    // d/dx(sin(x)) = cos(x)
    let s1 = format!("{}", dm.get(0, 1));
    assert_eq!(s1, "cos(x)", "d/dx(sin(x)) should be cos(x), got: {s1}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Lambdify
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lambdify_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(2) + &x * 3 + 1;
    let func = f.compile(&["x"]).expect("should compile");
    assert!((func(&[2.0]) - 11.0).abs() < 1e-10);
}

#[test]
fn lambdify_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin();
    let func = f.compile(&["x"]).expect("should compile");
    assert!((func(&[0.0])).abs() < 1e-10);
    assert!((func(&[std::f64::consts::FRAC_PI_2]) - 1.0).abs() < 1e-10);
}

#[test]
fn lambdify_two_vars() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x * &y + 1;
    let func = f.compile(&["x", "y"]).expect("should compile");
    assert!((func(&[3.0, 4.0]) - 13.0).abs() < 1e-10);
}

#[test]
fn lambdify_consistency_with_evalf() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.sin().powi(2) + &x.cos().powi(2);
    let func = f.compile(&["x"]).expect("should compile");
    // sin²+cos² should be 1 at any point
    assert!((func(&[1.5]) - 1.0).abs() < 1e-10);
    assert!((func(&[0.0]) - 1.0).abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// CSE
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cse_extracts_common() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();
    let expr = &sin_x.powi(2) + &sin_x;
    let (bindings, result) = expr.cse();
    // Verify the result formats without panic
    let result_s = format!("{result}");
    // sin(x) appears twice — CSE should extract at least one binding
    assert!(
        !bindings.is_empty(),
        "CSE should extract shared sin(x) subexpression"
    );
    // Verify one of the bindings contains sin(x)
    let any_sin = bindings
        .iter()
        .any(|(_, val)| format!("{val}").contains("sin"));
    assert!(
        any_sin,
        "at least one CSE binding should contain sin, bindings: {:?}",
        bindings
            .iter()
            .map(|(name, val)| format!("{name} = {val}"))
            .collect::<Vec<_>>()
    );
    // The result expression should reference the extracted binding variable(s),
    // not repeat sin(x) literally
    let result_sin_count = result_s.matches("sin").count();
    assert!(
        result_sin_count <= 1,
        "CSE result should not repeat sin(x) — got {result_sin_count} occurrences in: {result_s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ODE — unit tests live in src/ode.rs (dsolve requires arena-level
// Derivative node construction via pub(crate) intern, which is not
// accessible from integration tests). The tests below verify the
// module is publicly accessible and the types are usable.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_module_accessible() {
    // Verify the ODE module and its types are publicly exported
    let _result_type_check = |r: symplex::ode::OdeResult| {
        let _ = r.solution;
        let _ = r.constants;
    };
}
