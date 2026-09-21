//! 0.2 solving campaign — systems: symbolic `linsolve`, algebraic
//! polynomial systems (`solve_system_ex`), and Newton iteration
//! (`solve_numeric_system`).  Every solution is verified by substitution.

use symplex::polysys::{
    LinearSolution, NewtonOpts, linsolve, linsolve_matrix, solve_numeric_system,
    solve_numeric_system_with, solve_system_ex,
};
use symplex::prelude::*;

/// Substitute a `(var, value)` assignment into `eq` and assert it vanishes
/// (exactly after `simplify`, or numerically at random parameter values).
fn assert_satisfies(eq: &Ex, assignment: &[(Ex, Ex)], params: &[(&Ex, i64)], label: &str) {
    let mut r = eq.clone();
    for (v, val) in assignment {
        r = r.subs(v, val);
    }
    let simplified = r.simplify();
    if simplified.is_zero_structural() {
        return;
    }
    let mut num = simplified.clone();
    for (p, val) in params {
        num = num.subs_i64(p, *val);
    }
    let Complex64 { re, im } = num
        .eval_complex64()
        .unwrap_or_else(|e| panic!("{label}: residual {simplified} not evaluable: {e}"));
    assert!(
        re.hypot(im) < 1e-9,
        "{label}: residual {simplified} = {re} + {im}i"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// linsolve
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn linsolve_unique_numeric() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let eqs = [
        &x + &y + &z - 6,
        &x * 2 - &y + &z - 3,
        &x - &y * 3 + &z * 2 - 1,
    ];
    let vars = [x.clone(), y.clone(), z.clone()];
    let sol = linsolve(&eqs, &vars).unwrap();
    assert!(sol.is_unique());
    let pairs = sol.pairs().unwrap().to_vec();
    for e in &eqs {
        assert_satisfies(e, &pairs, &[], "3x3");
    }
    // Known answer: x = 1, y = 2, z = 3
    assert_eq!(format!("{}", sol.get(&x).unwrap()), "1");
    assert_eq!(format!("{}", sol.get(&y).unwrap()), "2");
    assert_eq!(format!("{}", sol.get(&z).unwrap()), "3");
}

#[test]
fn linsolve_symbolic_coefficients() {
    let ctx = Context::new();
    let (x, y, a, b) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("a"),
        ctx.symbol("b"),
    );
    // a x + y = 1,  x - y = b  →  x = (1+b)/(a+1), y = (1 - a b)/(a+1)
    let eqs = [&a * &x + &y - 1, &x - &y - &b];
    let vars = [x.clone(), y.clone()];
    let sol = linsolve(&eqs, &vars).unwrap();
    assert!(sol.is_unique(), "{sol:?}");
    let pairs = sol.pairs().unwrap().to_vec();
    for e in &eqs {
        assert_satisfies(e, &pairs, &[(&a, 3), (&b, 5)], "symbolic 2x2");
    }
    let xv = sol.get(&x).unwrap();
    let check = (&xv * &(&a + 1) - &(&b + 1)).simplify();
    assert!(check.is_zero_structural(), "x = {xv}");
    let yv = sol.get(&y).unwrap();
    let check = (&yv * &(&a + 1) - &(&ctx.int(1) - &(&a * &b))).simplify();
    assert!(check.is_zero_structural(), "y = {yv}");
}

#[test]
fn linsolve_underdetermined_parametric() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let eqs = [&x + &y + &z - 6, &x - &y + 1];
    let vars = [x.clone(), y.clone(), z.clone()];
    match linsolve(&eqs, &vars).unwrap() {
        LinearSolution::Parametric { solution, free } => {
            assert_eq!(free.len(), 1);
            assert_eq!(free[0], z);
            // z maps to itself; x, y are expressed in z.
            assert_eq!(solution[2].1, z);
            assert!(solution[0].1.contains(&z) && solution[1].1.contains(&z));
            for e in &eqs {
                assert_satisfies(e, &solution, &[(&z, 7)], "parametric");
            }
        }
        other => panic!("expected parametric, got {other:?}"),
    }
}

#[test]
fn linsolve_overdetermined_and_inconsistent() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // Consistent over-determined.
    let sol = linsolve(
        &[&x + &y - 3, &x - &y - 1, &x * 2 - 4],
        &[x.clone(), y.clone()],
    )
    .unwrap();
    assert!(sol.is_unique());
    assert_eq!(format!("{}", sol.get(&x).unwrap()), "2");
    // Inconsistent.
    let sol = linsolve(&[&x + &y - 3, &x + &y - 4], &[x.clone(), y.clone()]).unwrap();
    assert!(sol.is_inconsistent());
    assert!(sol.pairs().is_none());
    // Symbolic inconsistency: x = a, x = a + 1
    let a = ctx.symbol("a");
    let sol = linsolve(&[&x - &a, &x - &a - 1], std::slice::from_ref(&x)).unwrap();
    assert!(sol.is_inconsistent());
}

#[test]
fn linsolve_accepts_equations_and_rejects_nonlinear() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let eqs = [
        Equation::new(&x * 2 + &y, ctx.int(5)),
        Equation::new(&x - &y, ctx.int(1)),
    ];
    let sol = linsolve(&eqs, &[x.clone(), y.clone()]).unwrap();
    assert_eq!(format!("{}", sol.get(&x).unwrap()), "2");
    assert_eq!(format!("{}", sol.get(&y).unwrap()), "1");
    let err = linsolve(&[&x * &y - 1], &[x.clone(), y.clone()]).unwrap_err();
    assert!(matches!(err, SymplexError::InvalidArgument { .. }));
    let err = linsolve(&[x.sin()], std::slice::from_ref(&x)).unwrap_err();
    assert!(matches!(err, SymplexError::InvalidArgument { .. }));
    let err = linsolve::<Ex>(&[], std::slice::from_ref(&x)).unwrap_err();
    assert!(matches!(err, SymplexError::InvalidArgument { .. }));
}

#[test]
fn linsolve_matrix_form() {
    let ctx = Context::new();
    let a = symplex::matrix![ctx, [2, 1], [1, 3]];
    let b = Matrix::col_vector(vec![ctx.int(3), ctx.int(5)]);
    match linsolve_matrix(&a, &b).unwrap() {
        LinearSolution::Unique(pairs) => {
            assert_eq!(format!("{}", pairs[0].1), "4/5");
            assert_eq!(format!("{}", pairs[1].1), "7/5");
        }
        other => panic!("{other:?}"),
    }
    // Rank-deficient: [[1, 2], [2, 4]] x = [3, 6] → parametric
    let a = symplex::matrix![ctx, [1, 2], [2, 4]];
    let b = Matrix::col_vector(vec![ctx.int(3), ctx.int(6)]);
    assert!(matches!(
        linsolve_matrix(&a, &b).unwrap(),
        LinearSolution::Parametric { .. }
    ));
    // Inconsistent: [[1, 2], [2, 4]] x = [3, 7]
    let b = Matrix::col_vector(vec![ctx.int(3), ctx.int(7)]);
    assert!(linsolve_matrix(&a, &b).unwrap().is_inconsistent());
    // Shape mismatch.
    let b = Matrix::col_vector(vec![ctx.int(3)]);
    assert!(linsolve_matrix(&a, &b).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial systems
// ═══════════════════════════════════════════════════════════════════════════

fn check_system(eqs: &[Ex], vars: &[Ex], expected: usize, label: &str) -> Vec<Vec<Ex>> {
    let sols = solve_system_ex(eqs, vars).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_eq!(sols.len(), expected, "{label}: {sols:?}");
    for s in &sols {
        assert_eq!(s.len(), vars.len());
        let assignment: Vec<(Ex, Ex)> = vars.iter().cloned().zip(s.iter().cloned()).collect();
        for e in eqs {
            assert_satisfies(e, &assignment, &[], label);
        }
    }
    sols
}

#[test]
fn polysys_circle_line() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let sols = check_system(
        &[&x.powi(2) + &y.powi(2) - 1, &y - &x],
        &[x.clone(), y.clone()],
        2,
        "circle ∩ line",
    );
    for s in &sols {
        assert_eq!(s[0], s[1]);
        let v = s[0].eval_f64().unwrap().abs();
        assert!((v - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert!(format!("{}", s[0]).contains("sqrt(2)"), "{}", s[0]);
    }
}

#[test]
fn polysys_sqrt2_chain() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let sols = check_system(
        &[&x.powi(2) - 2, &y - &x - 1],
        &[x.clone(), y.clone()],
        2,
        "x^2 = 2, y = x + 1",
    );
    let strs: Vec<String> = sols.iter().map(|s| format!("{}", s[1])).collect();
    assert!(strs.iter().any(|s| s == "sqrt(2) + 1"), "{strs:?}");
    assert!(strs.iter().any(|s| s == "-sqrt(2) + 1"), "{strs:?}");
}

#[test]
fn polysys_golden_ratio_like() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let sols = check_system(
        &[&x * &y - 1, &x + &y - 3],
        &[x.clone(), y.clone()],
        2,
        "xy = 1, x + y = 3",
    );
    for s in &sols {
        assert!(format!("{}", s[0]).contains("sqrt(5)"), "{}", s[0]);
    }
}

#[test]
fn polysys_mixed_rational_and_irrational() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // x² + y² = 4, x² − y = 2 → (0, −2), (±√3, 1)
    let sols = check_system(
        &[&x.powi(2) + &y.powi(2) - 4, &x.powi(2) - &y - 2],
        &[x.clone(), y.clone()],
        3,
        "circle ∩ parabola",
    );
    let ys: Vec<String> = sols.iter().map(|s| format!("{}", s[1])).collect();
    assert_eq!(ys.iter().filter(|s| *s == "1").count(), 2, "{ys:?}");
    assert_eq!(ys.iter().filter(|s| *s == "-2").count(), 1, "{ys:?}");
}

#[test]
fn polysys_three_variables() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    // Elementary symmetric functions of the roots of t³ − 3t: {0, ±√3}
    check_system(
        &[&x + &y + &z, &x * &y + &y * &z + &z * &x + 3, &x * &y * &z],
        &[x.clone(), y.clone(), z.clone()],
        6,
        "symmetric 3-var",
    );
    // Triangular chain
    let sols = check_system(
        &[&x.powi(2) - 2, &y - &x - 1, &z - &x * &y],
        &[x.clone(), y.clone(), z.clone()],
        2,
        "3-var chain",
    );
    for s in &sols {
        // z = x(x+1) = 2 + x
        let diff = (&s[2] - &(&s[0] + 2)).simplify();
        assert!(diff.is_zero_structural(), "z = {}", s[2]);
    }
}

#[test]
fn polysys_complex_and_quartic_roots() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // x² = y, y² = 2 → y = ±√2, x = ±2^(1/4), ±i·2^(1/4)
    let sols = check_system(
        &[&x.powi(2) - &y, &y.powi(2) - 2],
        &[x.clone(), y.clone()],
        4,
        "x^2 = y, y^2 = 2",
    );
    let complex = sols.iter().filter(|s| s[0].contains(&ctx.i_unit())).count();
    assert_eq!(complex, 2);
}

#[test]
fn polysys_linear_delegates_to_linsolve() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let sols = check_system(
        &[&x + &y - 3, &x - &y - 1],
        &[x.clone(), y.clone()],
        1,
        "linear",
    );
    assert_eq!(format!("{}", sols[0][0]), "2");
    assert_eq!(format!("{}", sols[0][1]), "1");
    // Inconsistent linear → no solutions
    let sols = solve_system_ex(&[&x + &y - 3, &x + &y - 4], &[x.clone(), y.clone()]).unwrap();
    assert!(sols.is_empty());
}

#[test]
fn polysys_degenerate_cases() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // Inconsistent nonlinear.
    let sols = solve_system_ex(&[&x.powi(2) - 1, &x - 2], std::slice::from_ref(&x)).unwrap();
    assert!(sols.is_empty());
    // Under-determined linear → InfiniteSolutions
    assert!(matches!(
        solve_system_ex(&[&x + &y - 1], &[x.clone(), y.clone()]),
        Err(SymplexError::InfiniteSolutions { .. })
    ));
    // Positive-dimensional nonlinear → InfiniteSolutions
    assert!(matches!(
        solve_system_ex(&[&x * &y], &[x.clone(), y.clone()]),
        Err(SymplexError::InfiniteSolutions { .. })
    ));
    // Not polynomial → ComputationFailed
    assert!(matches!(
        solve_system_ex(&[&x.sin() - &y, &y - 1], &[x.clone(), y.clone()]),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// Newton
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn newton_circle_line() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let eqs = [&x.powi(2) + &y.powi(2) - 1, &y - &x];
    let vars = [x.clone(), y.clone()];
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let sol = solve_numeric_system(&eqs, &vars, &[1.0, 1.0]).unwrap();
    assert!((sol[0] - r).abs() < 1e-10 && (sol[1] - r).abs() < 1e-10);
    let sol = solve_numeric_system(&eqs, &vars, &[-1.0, -0.5]).unwrap();
    assert!((sol[0] + r).abs() < 1e-10 && (sol[1] + r).abs() < 1e-10);
    // Residual check by substitution.
    for e in &eqs {
        let v = e
            .subs(&x, &ctx.rational((sol[0] * 1e9) as i64, 1_000_000_000))
            .subs(&y, &ctx.rational((sol[1] * 1e9) as i64, 1_000_000_000))
            .eval_f64()
            .unwrap();
        assert!(v.abs() < 1e-8);
    }
}

#[test]
fn newton_transcendental_and_options() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // x = cos(y), y = sin(x)  — fixed point near (0.77, 0.69)
    let eqs = [&x - &y.cos(), &y - &x.sin()];
    let vars = [x.clone(), y.clone()];
    let opts = NewtonOpts {
        tol: 1e-13,
        max_iter: 200,
        damping: true,
    };
    let sol = solve_numeric_system_with(&eqs, &vars, &[1.0, 1.0], &opts).unwrap();
    assert!((sol[0] - sol[1].cos()).abs() < 1e-12);
    assert!((sol[1] - sol[0].sin()).abs() < 1e-12);
    // Undamped Newton also converges from a decent start.
    let plain = NewtonOpts {
        damping: false,
        ..NewtonOpts::default()
    };
    let sol2 = solve_numeric_system_with(&eqs, &vars, &[0.8, 0.7], &plain).unwrap();
    assert!((sol2[0] - sol[0]).abs() < 1e-9);
}

#[test]
fn newton_errors() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // Not square.
    assert!(matches!(
        solve_numeric_system(&[&x - 1], &[x.clone(), y.clone()], &[0.0, 0.0]),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // Wrong x0 length.
    assert!(matches!(
        solve_numeric_system(&[&x - 1, &y - 2], &[x.clone(), y.clone()], &[0.0]),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // No real solution: x² + 1 = 0, y = 0 → non-convergence with residual.
    let err = solve_numeric_system_with(
        &[&x.powi(2) + 1, y.clone()],
        &[x.clone(), y.clone()],
        &[1.0, 0.0],
        &NewtonOpts {
            max_iter: 30,
            ..NewtonOpts::default()
        },
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("residual") || msg.contains("singular"),
        "{msg}"
    );
}
