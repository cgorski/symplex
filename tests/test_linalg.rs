//! Public API tests for Context::solve_system().

use symplex::prelude::*;

#[test]
fn solve_system_2x2() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // x + y = 3, x - y = 1 → x=2, y=1
    let eq1 = &x + &y - 3;
    let eq2 = &x - &y - 1;
    let solution = ctx.solve_system(&[eq1, eq2], &[x, y]).unwrap();
    assert_eq!(solution.len(), 2);
    assert_eq!(format!("{}", solution[0].1), "2");
    assert_eq!(format!("{}", solution[1].1), "1");
}

#[test]
fn solve_system_with_coefficients() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // 2x + 3y = 7, x - y = 1 → x=2, y=1
    let eq1 = &x * 2 + &y * 3 - 7;
    let eq2 = &x - &y - 1;
    let solution = ctx.solve_system(&[eq1, eq2], &[x, y]).unwrap();
    assert_eq!(format!("{}", solution[0].1), "2");
    assert_eq!(format!("{}", solution[1].1), "1");
}

#[test]
fn solve_system_single_equation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 2x = 6 → x=3
    let eq = &x * 2 - 6;
    let solution = ctx.solve_system(&[eq], &[x]).unwrap();
    assert_eq!(format!("{}", solution[0].1), "3");
}

#[test]
fn solve_system_inconsistent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x = 1 AND x = 2 → inconsistent
    let eq1 = &x - 1;
    let eq2 = &x - 2;
    assert!(ctx.solve_system(&[eq1, eq2], &[x]).is_none());
}

#[test]
fn solve_system_rational_solution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 3x = 1 → x = 1/3
    let eq = &x * 3 - 1;
    let solution = ctx.solve_system(&[eq], &[x]).unwrap();
    assert_eq!(format!("{}", solution[0].1), "1/3");
}

#[test]
fn solve_system_3x3() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    // x + y + z = 6
    // x - y + z = 2
    // x + y - z = 0
    // Solution: x=1, y=2, z=3
    let eq1 = &x + &y + &z - 6;
    let eq2 = &x - &y + &z - 2;
    let eq3 = &x + &y - &z;
    let solution = ctx.solve_system(&[eq1, eq2, eq3], &[x, y, z]).unwrap();
    assert_eq!(format!("{}", solution[0].1), "1");
    assert_eq!(format!("{}", solution[1].1), "2");
    assert_eq!(format!("{}", solution[2].1), "3");
}
