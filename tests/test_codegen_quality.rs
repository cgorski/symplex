//! Quality tests for codegen output: constant propagation, subtraction style,
//! decimal fractions, zero elimination, and structural validity.

use symplex::matrix::jacobian;
use symplex::prelude::*;
use symplex::robotics::fk_position;

/// Build a 2-DOF planar robot Jacobian and verify no trivial constant temps.
#[test]
fn codegen_2dof_no_trivial_temps() {
    let ctx = Context::new();
    let theta1 = ctx.symbol("theta1");
    let theta2 = ctx.symbol("theta2");
    let l1 = ctx.symbol("L1");
    let l2 = ctx.symbol("L2");
    let zero = ctx.int(0);

    let dh: [(&Ex, &Ex, &Ex, &Ex); 2] =
        [(&theta1, &zero, &l1, &zero), (&theta2, &zero, &l2, &zero)];

    let (px, py, _pz) = fk_position(&dh);
    let jac = jacobian(&[&px, &py], &[&theta1, &theta2]);
    let code = jac
        .to_rust_fn("jacobian_2dof", &["theta1", "theta2", "L1", "L2"])
        .expect("codegen should succeed for 2-DOF Jacobian");

    // No trivial constant temporaries should appear
    assert!(
        !code.contains("= 1.0_f64;"),
        "should not emit `let tN = 1.0_f64;` binding in:\n{code}"
    );
    assert!(
        !code.contains("0.0_f64.powi"),
        "should not emit `0.0_f64.powi(...)` in:\n{code}"
    );
    assert!(
        !code.contains("= 0.0_f64;"),
        "should not emit `let tN = 0.0_f64;` binding in:\n{code}"
    );
}

/// Build a 3-DOF planar robot Jacobian and verify no trivial constant temps.
#[test]
fn codegen_3dof_no_trivial_temps() {
    let ctx = Context::new();
    let theta1 = ctx.symbol("theta1");
    let theta2 = ctx.symbol("theta2");
    let theta3 = ctx.symbol("theta3");
    let l1 = ctx.rational(3, 10);
    let l2 = ctx.rational(1, 4);
    let l3 = ctx.rational(1, 5);
    let zero = ctx.int(0);

    let dh: [(&Ex, &Ex, &Ex, &Ex); 3] = [
        (&theta1, &zero, &l1, &zero),
        (&theta2, &zero, &l2, &zero),
        (&theta3, &zero, &l3, &zero),
    ];

    let (px, py, _pz) = fk_position(&dh);
    let jac = jacobian(&[&px, &py], &[&theta1, &theta2, &theta3]);
    let code = jac
        .to_rust_fn("jacobian_3dof", &["theta1", "theta2", "theta3"])
        .expect("codegen should succeed for 3-DOF Jacobian");

    assert!(
        !code.contains("= 1.0_f64;"),
        "should not emit `let tN = 1.0_f64;` binding in:\n{code}"
    );
    assert!(
        !code.contains("0.0_f64.powi"),
        "should not emit `0.0_f64.powi(...)` in:\n{code}"
    );
    assert!(
        !code.contains("= 0.0_f64;"),
        "should not emit `let tN = 0.0_f64;` binding in:\n{code}"
    );
}

/// Subtraction style: `x + Neg(y)` should emit `- y`, not `+ (-y)`.
#[test]
fn codegen_subtraction_style() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // Build x + (-y) explicitly
    let neg_y = -&y;
    let expr = &x + &neg_y;

    let code = expr
        .to_rust_fn("sub_test", &["x", "y"])
        .expect("codegen should succeed");

    assert!(
        code.contains("- "),
        "expected subtraction operator `- ` in:\n{code}"
    );
    assert!(
        !code.contains("+ (-"),
        "should not contain `+ (-` pattern in:\n{code}"
    );
}

/// Fractions should be emitted as decimals, not as `N_f64 / M_f64`.
#[test]
fn codegen_fraction_as_decimal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let quarter = ctx.rational(1, 4);
    let expr = &quarter * &x;

    let code = expr
        .to_rust_fn("frac_test", &["x"])
        .expect("codegen should succeed");

    assert!(code.contains("0.25"), "expected decimal `0.25` in:\n{code}");
    assert!(
        !code.contains("1_f64 / 4_f64"),
        "should not contain fraction syntax `1_f64 / 4_f64` in:\n{code}"
    );
}

/// Zero elimination: `0 * sin(x)` should produce `0.0_f64` or similar.
#[test]
fn codegen_zero_elimination() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let expr = &zero * &x.sin();

    let code = expr
        .to_rust_fn("zero_test", &["x"])
        .expect("codegen should succeed");

    // The expression should simplify or constant-fold to something with 0.0
    // and should NOT contain a .sin() call since the product is zero
    assert!(
        code.contains("0.0") || code.contains("0_f64"),
        "expected zero result in:\n{code}"
    );
    // After canonicalization, 0*sin(x) should become just 0 — no sin call
    assert!(
        !code.contains(".sin()"),
        "should not contain sin() call for zero product in:\n{code}"
    );
}

/// 3-DOF Jacobian code should have balanced braces and valid structure.
#[test]
fn codegen_3dof_compiles() {
    let ctx = Context::new();
    let theta1 = ctx.symbol("theta1");
    let theta2 = ctx.symbol("theta2");
    let theta3 = ctx.symbol("theta3");
    let l1 = ctx.rational(3, 10);
    let l2 = ctx.rational(1, 4);
    let l3 = ctx.rational(1, 5);
    let zero = ctx.int(0);

    let dh: [(&Ex, &Ex, &Ex, &Ex); 3] = [
        (&theta1, &zero, &l1, &zero),
        (&theta2, &zero, &l2, &zero),
        (&theta3, &zero, &l3, &zero),
    ];

    let (px, py, _pz) = fk_position(&dh);
    let jac = jacobian(&[&px, &py], &[&theta1, &theta2, &theta3]);
    let code = jac
        .to_rust_fn("robot_jac", &["theta1", "theta2", "theta3"])
        .expect("codegen should succeed for 3-DOF Jacobian");

    // Balanced braces
    let open_braces = code.chars().filter(|&c| c == '{').count();
    let close_braces = code.chars().filter(|&c| c == '}').count();
    assert_eq!(
        open_braces, close_braces,
        "unbalanced braces ({open_braces} open vs {close_braces} close) in:\n{code}"
    );

    // Balanced parentheses
    let open_parens = code.chars().filter(|&c| c == '(').count();
    let close_parens = code.chars().filter(|&c| c == ')').count();
    assert_eq!(
        open_parens, close_parens,
        "unbalanced parens ({open_parens} open vs {close_parens} close) in:\n{code}"
    );

    // Balanced brackets
    let open_brackets = code.chars().filter(|&c| c == '[').count();
    let close_brackets = code.chars().filter(|&c| c == ']').count();
    assert_eq!(
        open_brackets, close_brackets,
        "unbalanced brackets ({open_brackets} open vs {close_brackets} close) in:\n{code}"
    );

    // Should contain a valid function signature
    assert!(
        code.contains("pub fn robot_jac("),
        "missing function signature in:\n{code}"
    );

    // Should contain a return type with array
    assert!(
        code.contains("[f64;"),
        "missing array return type in:\n{code}"
    );
}
