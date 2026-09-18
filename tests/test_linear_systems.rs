//! Tests for linear system solving through `Context::solve_system` /
//! `linsolve` with a variety of system shapes: 2×2, 3×3, inconsistent,
//! underdetermined (parametric), rational coefficients, negative/zero
//! solutions, and substitution-based verification.

use symplex::prelude::*;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Unwrap a unique solution and format the value side of each pair.
fn unique_values(sol: LinearSolution) -> Vec<String> {
    match sol {
        LinearSolution::Unique(pairs) => pairs.iter().map(|(_, v)| format!("{v}")).collect(),
        other => panic!("expected a unique solution, got {other:?}"),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn solve_2x2_basic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x + y = 3, x - y = 1  →  x = 2, y = 1
    let vals = unique_values(
        ctx.solve_system(&[&x + &y - 3, &x - &y - 1], &[x, y])
            .unwrap(),
    );
    assert_eq!(vals, ["2", "1"]);
}

#[test]
fn solve_3x3_system() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    // x + y + z = 6, 2x + y - z = 1, x - y + z = 2  →  x = 1, y = 2, z = 3
    let eq1 = &x + &y + &z - 6;
    let eq2 = &x * 2 + &y - &z - 1;
    let eq3 = &x - &y + &z - 2;
    let vals = unique_values(ctx.solve_system(&[eq1, eq2, eq3], &[x, y, z]).unwrap());
    assert_eq!(vals, ["1", "2", "3"]);
}

#[test]
fn solve_inconsistent_system() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x + y = 1  and  x + y = 2  — no solution
    let sol = ctx
        .solve_system(&[&x + &y - 1, &x + &y - 2], &[x, y])
        .unwrap();
    assert!(
        matches!(sol, LinearSolution::Inconsistent),
        "inconsistent system should be reported as such, got {sol:?}"
    );
    assert!(sol.is_inconsistent());
    assert!(!sol.is_unique());
}

#[test]
fn solve_underdetermined_system_is_parametric() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // One equation, two unknowns: x + y = 5  →  x = 5 − y, y free.
    let sol = ctx
        .solve_system(&[&x + &y - 5], &[x.clone(), y.clone()])
        .unwrap();
    match &sol {
        LinearSolution::Parametric { solution, free } => {
            assert_eq!(free.len(), 1, "exactly one free variable: {free:?}");
            let free_var = &free[0];
            // The pinned variable is expressed in terms of the free one and
            // satisfies the equation identically.
            let (pinned, value) = solution
                .iter()
                .find(|(v, _)| v != free_var)
                .expect("a pinned variable");
            let residual = (&x + &y - 5).subs(pinned, value).simplify();
            assert_eq!(format!("{residual}"), "0", "residual: {residual}");
        }
        other => panic!("expected Parametric, got {other:?}"),
    }
}

#[test]
fn solve_rational_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // 3x + 2y = 1, x - y = 2  →  x = 1, y = -1
    let vals = unique_values(
        ctx.solve_system(&[&x * 3 + &y * 2 - 1, &x - &y - 2], &[x, y])
            .unwrap(),
    );
    assert_eq!(vals, ["1", "-1"]);
}

#[test]
fn solve_rational_solution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x + y = 1, 2x - y = 0  →  x = 1/3, y = 2/3
    let vals = unique_values(
        ctx.solve_system(&[&x + &y - 1, &x * 2 - &y], &[x, y])
            .unwrap(),
    );
    assert_eq!(vals, ["1/3", "2/3"]);
}

#[test]
fn solve_single_equation_single_variable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 5x - 10 = 0  →  x = 2
    let vals = unique_values(ctx.solve_system(&[&x * 5 - 10], &[x]).unwrap());
    assert_eq!(vals, ["2"]);
}

#[test]
fn solve_negative_solutions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x + y = -3, x - y = 1  →  x = -1, y = -2
    let vals = unique_values(
        ctx.solve_system(&[&x + &y + 3, &x - &y - 1], &[x, y])
            .unwrap(),
    );
    assert_eq!(vals, ["-1", "-2"]);
}

#[test]
fn solve_zero_solution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x + y = 0, x - y = 0  →  x = 0, y = 0
    let vals = unique_values(ctx.solve_system(&[&x + &y, &x - &y], &[x, y]).unwrap());
    assert_eq!(vals, ["0", "0"]);
}

#[test]
fn solve_verify_by_substitution() {
    // Solve, then substitute the solution back into the original equations
    // and confirm they evaluate to zero.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // 2x + 3y = 8, x - 2y = -3
    let eq1 = &x * 2 + &y * 3 - 8;
    let eq2 = &x - &y * 2 + 3;

    let pairs = match ctx
        .solve_system(&[eq1.clone(), eq2.clone()], &[x.clone(), y.clone()])
        .unwrap()
    {
        LinearSolution::Unique(p) => p,
        other => panic!("expected unique, got {other:?}"),
    };
    assert_eq!(pairs.len(), 2);

    let (_, x_val) = &pairs[0];
    let (_, y_val) = &pairs[1];
    let check1 = eq1.subs(&x, x_val).subs(&y, y_val).eval();
    let check2 = eq2.subs(&x, x_val).subs(&y, y_val).eval();
    assert_eq!(format!("{check1}"), "0", "eq1 should vanish at solution");
    assert_eq!(format!("{check2}"), "0", "eq2 should vanish at solution");
}

#[test]
fn solve_larger_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // 10x + 7y = 41, 3x - 5y = -9  →  x = 2, y = 3
    let vals = unique_values(
        ctx.solve_system(&[&x * 10 + &y * 7 - 41, &x * 3 - &y * 5 + 9], &[x, y])
            .unwrap(),
    );
    assert_eq!(vals, ["2", "3"]);
}

#[test]
fn solve_redundant_equations_still_unique() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // Three equations, one redundant (eq3 = eq1 + eq2): still a unique solution.
    let eq1 = &x + &y - 3;
    let eq2 = &x - &y - 1;
    let eq3 = &x * 2 - 4;
    let vals = unique_values(ctx.solve_system(&[eq1, eq2, eq3], &[x, y]).unwrap());
    assert_eq!(vals, ["2", "1"]);
}

#[test]
fn solve_empty_equations_is_an_error() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let empty: [Ex; 0] = [];
    let err = ctx.solve_system(&empty, &[x]).unwrap_err();
    assert!(
        matches!(err, SymplexError::InvalidArgument { .. }),
        "empty equation list should be InvalidArgument, got {err}"
    );
}

#[test]
fn solve_empty_variables_is_an_error() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let err = ctx.solve_system(&[&x - 1], &[]).unwrap_err();
    assert!(
        matches!(err, SymplexError::InvalidArgument { .. }),
        "empty variable list should be InvalidArgument, got {err}"
    );
}

#[test]
fn solve_nonlinear_input_is_rejected() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let err = ctx
        .solve_system(&[&x * &y - 1, &x - &y], &[x, y])
        .unwrap_err();
    assert!(
        matches!(err, SymplexError::InvalidArgument { .. }),
        "non-linear system should be InvalidArgument, got {err}"
    );
}
