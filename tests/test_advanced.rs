//! Integration tests for advanced features (Cycles 16-18).

// ═══════════════════════════════════════════════════════════════════════════
// Matrix
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_identity_times_vector() {
    use symplex::matrix::Matrix;
    let x = symplex::var("x");
    let y = symplex::var("y");
    let id = Matrix::identity(2);
    let v = Matrix::col_vector(vec![x.clone(), y.clone()]);
    let result = id.matmul(&v);
    assert_eq!(format!("{}", result.get(0, 0)), "x");
    assert_eq!(format!("{}", result.get(1, 0)), "y");
}

#[test]
fn matrix_det_2x2_numeric() {
    use symplex::matrix::Matrix;
    let m = Matrix::new(vec![
        vec![symplex::int(3), symplex::int(7)],
        vec![symplex::int(1), symplex::int(5)],
    ]);
    let det = m.det();
    // 3*5 - 7*1 = 8
    assert_eq!(format!("{det}"), "8");
}

#[test]
fn matrix_jacobian() {
    use symplex::matrix::jacobian;
    let x = symplex::var("x");
    let y = symplex::var("y");
    let f1 = &x.powi(2) * &y;
    let f2 = &x + &y.powi(3);
    let j = jacobian(&[f1, f2], &[x.clone(), y.clone()]);
    assert_eq!(j.nrows(), 2);
    assert_eq!(j.ncols(), 2);
}

#[test]
fn matrix_trace() {
    use symplex::matrix::Matrix;
    let m = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(3), symplex::int(4)],
    ]);
    let tr = m.trace();
    assert_eq!(format!("{tr}"), "5");
}

#[test]
fn matrix_add_numeric() {
    use symplex::matrix::Matrix;
    let m1 = Matrix::new(vec![
        vec![symplex::int(1), symplex::int(2)],
        vec![symplex::int(3), symplex::int(4)],
    ]);
    let m2 = Matrix::new(vec![
        vec![symplex::int(10), symplex::int(20)],
        vec![symplex::int(30), symplex::int(40)],
    ]);
    let sum = m1.add(&m2);
    assert_eq!(format!("{}", sum.get(0, 0)), "11");
    assert_eq!(format!("{}", sum.get(1, 1)), "44");
}

#[test]
fn matrix_diff() {
    use symplex::matrix::Matrix;
    let x = symplex::var("x");
    let m = Matrix::new(vec![vec![x.powi(2), x.sin()]]);
    let dm = m.diff(&x);
    let s = format!("{}", dm.get(0, 0));
    assert!(s.contains("x"), "d/dx(x²) should contain x: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Lambdify
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lambdify_polynomial() {
    let x = symplex::var("x");
    let f = &x.powi(2) + &x * 3 + 1;
    let func = f.lambdify(&["x"]).expect("should compile");
    assert!((func(&[2.0]) - 11.0).abs() < 1e-10);
}

#[test]
fn lambdify_trig() {
    let x = symplex::var("x");
    let f = x.sin();
    let func = f.lambdify(&["x"]).expect("should compile");
    assert!((func(&[0.0])).abs() < 1e-10);
    assert!((func(&[std::f64::consts::FRAC_PI_2]) - 1.0).abs() < 1e-10);
}

#[test]
fn lambdify_two_vars() {
    let x = symplex::var("x");
    let y = symplex::var("y");
    let f = &x * &y + 1;
    let func = f.lambdify(&["x", "y"]).expect("should compile");
    assert!((func(&[3.0, 4.0]) - 13.0).abs() < 1e-10);
}

#[test]
fn lambdify_consistency_with_evalf() {
    let x = symplex::var("x");
    let f = &x.sin().powi(2) + &x.cos().powi(2);
    let func = f.lambdify(&["x"]).expect("should compile");
    // sin²+cos² should be 1 at any point
    assert!((func(&[1.5]) - 1.0).abs() < 1e-10);
    assert!((func(&[0.0]) - 1.0).abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// CSE
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cse_extracts_common() {
    let x = symplex::var("x");
    let sin_x = x.sin();
    let expr = &sin_x.powi(2) + &sin_x;
    let (bindings, result) = expr.cse();
    // Verify the result formats without panic
    let _ = format!("{result}");
    // sin(x) appears twice — CSE should extract at least one binding
    // (canonicalization may affect exact count, so just verify non-empty)
    assert!(
        !bindings.is_empty(),
        "CSE should extract shared sin(x) subexpression"
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
