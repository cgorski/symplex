//! Tests for linear system solving (Gaussian elimination).
//!
//! Exercises `Context::solve_system` through the public API with a variety
//! of system shapes: 2×2, 3×3, inconsistent, underdetermined, rational
//! coefficients, negative/zero solutions, and substitution-based verification.

use symplex::prelude::*;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Format the value side of every solution pair into a `Vec<String>`.
fn solution_values(pairs: &[(Ex, Ex)]) -> Vec<String> {
    pairs.iter().map(|(_, v)| format!("{v}")).collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn solve_2x2_basic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x + y = 3, x - y = 1  →  x = 2, y = 1
    let eq1 = &x + &y - 3; // x + y - 3 = 0
    let eq2 = &x - &y - 1; // x - y - 1 = 0
    let result = ctx.solve_system(&[eq1, eq2], &[x, y]);
    assert!(result.is_some(), "2×2 system should have a unique solution");
    let sol = result.unwrap();
    assert_eq!(sol.len(), 2);
    let vals = solution_values(&sol);
    assert_eq!(vals[0], "2", "x should be 2");
    assert_eq!(vals[1], "1", "y should be 1");
}

#[test]
fn solve_3x3_system() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    // x + y + z = 6    →  x = 1, y = 2, z = 3
    // 2x + y - z = 1
    // x - y + z = 2
    let eq1 = &x + &y + &z - 6;
    let eq2 = &x * 2 + &y - &z - 1;
    let eq3 = &x - &y + &z - 2;
    let result = ctx.solve_system(&[eq1, eq2, eq3], &[x, y, z]);
    assert!(result.is_some(), "3×3 system should have a unique solution");
    let vals = solution_values(&result.unwrap());
    assert_eq!(vals[0], "1");
    assert_eq!(vals[1], "2");
    assert_eq!(vals[2], "3");
}

#[test]
fn solve_inconsistent_system() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x + y = 1  and  x + y = 2  — no solution
    let eq1 = &x + &y - 1;
    let eq2 = &x + &y - 2;
    let result = ctx.solve_system(&[eq1, eq2], &[x, y]);
    assert!(result.is_none(), "inconsistent system should return None");
}

#[test]
fn solve_underdetermined_system() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // One equation, two unknowns → underdetermined → None
    let eq1 = &x + &y - 5;
    let result = ctx.solve_system(&[eq1], &[x, y]);
    assert!(
        result.is_none(),
        "underdetermined system (1 eq, 2 vars) should return None"
    );
}

#[test]
fn solve_rational_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // 3x + 2y = 1   →  x = 1, y = -1
    // x - y   = 2
    let eq1 = &x * 3 + &y * 2 - 1;
    let eq2 = &x - &y - 2;
    let result = ctx.solve_system(&[eq1, eq2], &[x, y]);
    assert!(
        result.is_some(),
        "system with integer coefficients should solve"
    );
    let vals = solution_values(&result.unwrap());
    assert_eq!(vals[0], "1");
    assert_eq!(vals[1], "-1");
}

#[test]
fn solve_rational_solution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x + y = 1   →  x = 1/3, y = 2/3
    // 2x - y = 0
    let eq1 = &x + &y - 1;
    let eq2 = &x * 2 - &y;
    let result = ctx.solve_system(&[eq1, eq2], &[x, y]);
    assert!(
        result.is_some(),
        "system with rational solution should solve"
    );
    let vals = solution_values(&result.unwrap());
    assert_eq!(vals[0], "1/3");
    assert_eq!(vals[1], "2/3");
}

#[test]
fn solve_single_equation_single_variable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 5x - 10 = 0  →  x = 2
    let eq = &x * 5 - 10;
    let result = ctx.solve_system(&[eq], &[x]);
    assert!(result.is_some());
    let vals = solution_values(&result.unwrap());
    assert_eq!(vals[0], "2");
}

#[test]
fn solve_negative_solutions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x + y = -3   →  x = -1, y = -2
    // x - y = 1
    let eq1 = &x + &y + 3; // x + y + 3 = 0  i.e. x + y = -3
    let eq2 = &x - &y - 1;
    let result = ctx.solve_system(&[eq1, eq2], &[x, y]);
    assert!(
        result.is_some(),
        "system with negative solutions should solve"
    );
    let vals = solution_values(&result.unwrap());
    assert_eq!(vals[0], "-1");
    assert_eq!(vals[1], "-2");
}

#[test]
fn solve_zero_solution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x + y = 0   →  x = 0, y = 0
    // x - y = 0
    let eq1 = &x + &y;
    let eq2 = &x - &y;
    let result = ctx.solve_system(&[eq1, eq2], &[x, y]);
    assert!(
        result.is_some(),
        "homogeneous system should have trivial solution"
    );
    let vals = solution_values(&result.unwrap());
    assert_eq!(vals[0], "0");
    assert_eq!(vals[1], "0");
}

#[test]
fn solve_verify_by_substitution() {
    // Build equations, solve, then substitute the solution back into the
    // original equations and confirm they evaluate to zero.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // Keep clones for substitution after solve consumes x, y.
    let x_ref = x.clone();
    let y_ref = y.clone();

    // 2x + 3y = 8   →  some solution
    // x  - 2y = -3
    let eq1 = &x * 2 + &y * 3 - 8;
    let eq2 = &x - &y * 2 + 3;

    // Preserve equation clones for later substitution.
    let eq1_check = &x_ref * 2 + &y_ref * 3 - 8;
    let eq2_check = &x_ref - &y_ref * 2 + 3;

    let result = ctx
        .solve_system(&[eq1, eq2], &[x, y])
        .expect("should solve");
    assert_eq!(result.len(), 2);

    // Substitute solution into both equations.
    let (_, x_val) = &result[0];
    let (_, y_val) = &result[1];

    let check1 = eq1_check.subs(&x_ref, x_val).subs(&y_ref, y_val);
    let check2 = eq2_check.subs(&x_ref, x_val).subs(&y_ref, y_val);

    assert_eq!(format!("{check1}"), "0", "eq1 should vanish at solution");
    assert_eq!(format!("{check2}"), "0", "eq2 should vanish at solution");
}

#[test]
fn solve_larger_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // 10x + 7y = 41    →  x = 2, y = 3
    //  3x - 5y = -9
    let eq1 = &x * 10 + &y * 7 - 41;
    let eq2 = &x * 3 - &y * 5 + 9;
    let result = ctx.solve_system(&[eq1, eq2], &[x, y]);
    assert!(result.is_some());
    let vals = solution_values(&result.unwrap());
    assert_eq!(vals[0], "2");
    assert_eq!(vals[1], "3");
}

#[test]
fn solve_empty_equations_returns_none() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.solve_system(&[], &[x]);
    assert!(result.is_none(), "empty equation list should return None");
}

#[test]
fn solve_empty_variables_returns_none() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x - 1;
    let result = ctx.solve_system(&[eq], &[]);
    assert!(result.is_none(), "empty variable list should return None");
}
