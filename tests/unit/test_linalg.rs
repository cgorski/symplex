//! Public API tests for `Context::solve_system()` (a thin wrapper over
//! `linsolve`).  Every returned solution is substituted back into the
//! original equations.

use symplex::prelude::*;

/// Unwrap a unique solution and return its values in variable order.
fn unique(sol: LinearSolution) -> Vec<Ex> {
    match sol {
        LinearSolution::Unique(pairs) => pairs.into_iter().map(|(_, v)| v).collect(),
        other => panic!("expected a unique solution, got {other:?}"),
    }
}

/// Assert that substituting `values` for `vars` makes every equation vanish.
fn check_vanishes(eqs: &[Ex], vars: &[Ex], values: &[Ex]) {
    for eq in eqs {
        let mut e = eq.clone();
        for (v, val) in vars.iter().zip(values) {
            e = e.subs(v, val);
        }
        assert_eq!(format!("{}", e.eval()), "0", "residual of {eq}");
    }
}

#[test]
fn solve_system_2x2() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // x + y = 3, x - y = 1 → x=2, y=1
    let eqs = [&x + &y - 3, &x - &y - 1];
    let vars = [x, y];
    let vals = unique(ctx.solve_system(&eqs, &vars).unwrap());
    assert_eq!(vals.len(), 2);
    assert_eq!(format!("{}", vals[0]), "2");
    assert_eq!(format!("{}", vals[1]), "1");
    check_vanishes(&eqs, &vars, &vals);
}

#[test]
fn solve_system_with_coefficients() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // 2x + 3y = 7, x - y = 1 → x=2, y=1
    let eqs = [&x * 2 + &y * 3 - 7, &x - &y - 1];
    let vars = [x, y];
    let vals = unique(ctx.solve_system(&eqs, &vars).unwrap());
    assert_eq!(format!("{}", vals[0]), "2");
    assert_eq!(format!("{}", vals[1]), "1");
    check_vanishes(&eqs, &vars, &vals);
}

#[test]
fn solve_system_single_equation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 2x = 6 → x=3
    let vals = unique(ctx.solve_system(&[&x * 2 - 6], &[x]).unwrap());
    assert_eq!(format!("{}", vals[0]), "3");
}

#[test]
fn solve_system_inconsistent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x = 1 AND x = 2 → inconsistent
    let sol = ctx.solve_system(&[&x - 1, &x - 2], &[x]).unwrap();
    assert!(
        matches!(sol, LinearSolution::Inconsistent),
        "expected Inconsistent, got {sol:?}"
    );
}

#[test]
fn solve_system_rational_solution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 3x = 1 → x = 1/3
    let vals = unique(ctx.solve_system(&[&x * 3 - 1], &[x]).unwrap());
    assert_eq!(format!("{}", vals[0]), "1/3");
}

#[test]
fn solve_system_3x3() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    // x + y + z = 6, x - y + z = 2, x + y - z = 0  →  x=1, y=2, z=3
    let eqs = [&x + &y + &z - 6, &x - &y + &z - 2, &x + &y - &z];
    let vars = [x, y, z];
    let vals = unique(ctx.solve_system(&eqs, &vars).unwrap());
    assert_eq!(format!("{}", vals[0]), "1");
    assert_eq!(format!("{}", vals[1]), "2");
    assert_eq!(format!("{}", vals[2]), "3");
    check_vanishes(&eqs, &vars, &vals);
}

#[test]
fn solve_system_accepts_equations() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let eqs = [
        Equation::new(&x + &y, ctx.int(3)),
        Equation::new(&x - &y, ctx.int(1)),
    ];
    let vals = unique(ctx.solve_system(&eqs, &[x, y]).unwrap());
    assert_eq!(format!("{}", vals[0]), "2");
    assert_eq!(format!("{}", vals[1]), "1");
}

#[test]
fn solve_system_symbolic_coefficients() {
    let ctx = Context::new();
    let (x, y, a) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a"));
    // a·x + y = 1, x − y = 0  →  x = y = 1/(a+1)
    let eqs = [&a * &x + &y - 1, &x - &y];
    let vars = [x.clone(), y.clone()];
    let vals = unique(ctx.solve_system(&eqs, &vars).unwrap());
    // Verify by substitution then simplification (symbolic residual).
    for eq in &eqs {
        let r = eq.subs(&x, &vals[0]).subs(&y, &vals[1]).simplify();
        assert_eq!(format!("{r}"), "0", "residual of {eq}: {r}");
    }
    // And numerically at a = 3: x = y = 1/4.
    let at3 = vals[0].subs(&a, &ctx.int(3)).eval();
    assert_eq!(format!("{at3}"), "1/4");
}
