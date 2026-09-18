//! Regression tests for the 0.2 "silently wrong solver results" campaign:
//! polynomial systems (`solve_system_ex`) and the univariate cubic /
//! quartic solvers they depend on.
//!
//! Root causes fixed:
//! * Cardano's formula emitted `cbrt(negative)`; `evalf` evaluates that on
//!   the principal complex branch, so every quartic whose resolvent cubic
//!   had a negative radicand produced wrong roots.
//! * Ferrari's factorisation divides by `k = √(2m − p)`, which is `0` for
//!   biquadratics (`q = 0`) — those are now solved as quadratics in `x²`.
//! * `solve_system_ex` never verified its tuples; it now checks every
//!   candidate against every original equation numerically.

use symplex::prelude::*;

/// Residual `|f(sol)|` of `f` at the solution tuple, using simultaneous
/// substitution (a `RootOf` value carries the solve variable as a bound
/// symbol, so sequential substitution would corrupt it).
fn residual(f: &Ex, vars: &[Ex], sol: &[Ex]) -> f64 {
    let pairs: Vec<(&Ex, &Ex)> = vars.iter().zip(sol.iter()).collect();
    let (re, im) = f
        .subs_map(&pairs)
        .eval_complex64()
        .unwrap_or_else(|e| panic!("cannot evaluate residual of {f} at {sol:?}: {e}"));
    re.hypot(im)
}

/// Solve, assert the expected number of solutions, and verify every
/// tuple against every equation by substitution.
fn check_system(eqs: &[Ex], vars: &[Ex], expected: usize) -> Vec<Vec<Ex>> {
    let sols = symplex::polysys::solve_system_ex(eqs, vars).expect("solvable system");
    assert_eq!(
        sols.len(),
        expected,
        "expected {expected} solutions, got {}: {sols:?}",
        sols.len()
    );
    for sol in &sols {
        assert_eq!(sol.len(), vars.len());
        for eq in eqs {
            let r = residual(eq, vars, sol);
            assert!(r < 1e-8, "residual {r} of {eq} at {sol:?}");
        }
    }
    // Solutions must be pairwise distinct numerically.
    let pts: Vec<Vec<(f64, f64)>> = sols
        .iter()
        .map(|s| s.iter().map(|v| v.eval_complex64().unwrap()).collect())
        .collect();
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            let same = pts[i]
                .iter()
                .zip(&pts[j])
                .all(|(a, b)| (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9);
            assert!(!same, "duplicate solution {:?}", pts[i]);
        }
    }
    sols
}

fn xy(ctx: &Context) -> (Ex, Ex) {
    (ctx.symbol("x"), ctx.symbol("y"))
}

// ── the two oracle reproducers ──────────────────────────────────────────

#[test]
fn quartic_resultant_system_has_four_verified_solutions() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    let sols = check_system(
        &[x.powi(2) + &y - 3, &x - y.powi(2) + 1],
        &[x.clone(), y.clone()],
        4,
    );
    // SymPy: (2.1875…, -1.7854…) is one of the two real solutions.
    let has_real = sols.iter().any(|s| {
        let (xv, yv) = (
            s[0].eval_complex64().unwrap(),
            s[1].eval_complex64().unwrap(),
        );
        (xv.0 - 2.18754904943214).abs() < 1e-9
            && xv.1.abs() < 1e-12
            && (yv.0 + 1.78537084367146).abs() < 1e-9
    });
    assert!(
        has_real,
        "missing the real solution (2.1875, -1.7854): {sols:?}"
    );
}

#[test]
fn biquadratic_eliminant_system_is_not_empty() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    let sols = check_system(
        &[&x * &y - 1, x.powi(2) - y.powi(2) - 3],
        &[x.clone(), y.clone()],
        4,
    );
    let reals = sols
        .iter()
        .filter(|s| s[0].eval_complex64().unwrap().1.abs() < 1e-12)
        .count();
    assert_eq!(reals, 2, "two real and two purely imaginary solutions");
}

// ── univariate cubic / quartic root correctness ────────────────────────

fn check_univariate_roots(p: &Ex, x: &Ex, expected: usize) -> Vec<Ex> {
    let roots = p.solve(x).expect("solvable");
    assert_eq!(roots.len(), expected, "roots of {p}: {roots:?}");
    for r in &roots {
        let (re, im) = p.subs(x, r).eval_complex64().unwrap();
        assert!(
            re.hypot(im) < 1e-9,
            "root {r} of {p} has residual {re}+{im}i"
        );
    }
    roots
}

#[test]
fn cubic_one_real_root_negative_cardano_radicand() {
    // m³ + m² + 2m + 15/8: Δ > 0, and −q/2 − √Δ < 0 (the branch that
    // used to be evaluated as a complex cube root).
    let ctx = Context::new();
    let m = ctx.symbol("m");
    let p = m.powi(3) + m.powi(2) + 2 * &m + ctx.rational(15, 8);
    let roots = check_univariate_roots(&p, &m, 3);
    let reals: Vec<f64> = roots
        .iter()
        .filter_map(|r| {
            let (re, im) = r.eval_complex64().unwrap();
            (im.abs() < 1e-12).then_some(re)
        })
        .collect();
    assert_eq!(reals.len(), 1);
    assert!((reals[0] + 0.957_134_627_407_467_7).abs() < 1e-9);
    // No cube root of a negative real appears in the printed roots.
    for r in &roots {
        assert!(
            !format!("{r}").contains("cbrt(-"),
            "negative cbrt radicand in {r}"
        );
    }
}

#[test]
fn cubic_one_real_root_positive_q_negative_p() {
    // x³ − 3x + 5: p = −3 < 0, q = 5 > 0 ⇒ both radicands negative.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = x.powi(3) - 3 * &x + 5;
    let roots = check_univariate_roots(&p, &x, 3);
    let real: Vec<_> = roots
        .iter()
        .filter(|r| r.eval_complex64().unwrap().1.abs() < 1e-12)
        .collect();
    assert_eq!(real.len(), 1);
    assert!((real[0].eval_f64().unwrap() + 2.279_018_786_).abs() < 1e-8);
}

#[test]
fn cubic_three_real_roots_casus_irreducibilis() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = x.powi(3) - 3 * &x + 1;
    let roots = check_univariate_roots(&p, &x, 3);
    for r in &roots {
        assert!(
            r.eval_complex64().unwrap().1.abs() < 1e-12,
            "{r} should be real"
        );
    }
}

#[test]
fn quartic_with_negative_resolvent_radicand() {
    let ctx = Context::new();
    let y = ctx.symbol("y");
    let p = y.powi(4) - 2 * y.powi(2) + &y - 2;
    let roots = check_univariate_roots(&p, &y, 4);
    let reals: Vec<f64> = roots
        .iter()
        .filter_map(|r| {
            let (re, im) = r.eval_complex64().unwrap();
            (im.abs() < 1e-12).then_some(re)
        })
        .collect();
    assert_eq!(reals.len(), 2);
    assert!(
        reals
            .iter()
            .any(|v| (v - 1.492_572_713_238_452).abs() < 1e-9)
    );
    assert!(
        reals
            .iter()
            .any(|v| (v + 1.785_370_843_671_46).abs() < 1e-9)
    );
}

#[test]
fn biquadratic_quartic_irrational() {
    // y⁴ + 3y² − 1: q = 0, resolvent root m = p/2 makes Ferrari's k = 0.
    let ctx = Context::new();
    let y = ctx.symbol("y");
    let p = y.powi(4) + 3 * y.powi(2) - 1;
    let roots = check_univariate_roots(&p, &y, 4);
    let reals = roots
        .iter()
        .filter(|r| r.eval_complex64().unwrap().1.abs() < 1e-12)
        .count();
    assert_eq!(reals, 2);
}

#[test]
fn biquadratic_quartic_all_complex() {
    // x⁴ + x² + 1 = 0: roots are the primitive 6th roots of unity.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = x.powi(4) + x.powi(2) + 1;
    let roots = check_univariate_roots(&p, &x, 4);
    for r in &roots {
        let (re, im) = r.eval_complex64().unwrap();
        assert!(
            (re.hypot(im) - 1.0).abs() < 1e-9,
            "{r} not on the unit circle"
        );
    }
}

#[test]
fn shifted_biquadratic_quartic() {
    // (x−1)⁴ + 3(x−1)² − 1 expanded: b ≠ 0 but the depressed q = 0.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let t = &x - 1;
    let p = (t.powi(4) + 3 * t.powi(2) - 1).expand();
    check_univariate_roots(&p, &x, 4);
}

#[test]
fn quartic_general_ferrari_cardano_resolvent() {
    // x⁴ + x³ − 2x − 3: no rational roots, resolvent needs Cardano.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = x.powi(4) + x.powi(3) - 2 * &x - 3;
    check_univariate_roots(&p, &x, 4);
}

// ── polynomial systems, each verified by substitution ──────────────────

#[test]
fn system_circle_and_hyperbola() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    check_system(
        &[x.powi(2) + y.powi(2) - 4, &x * &y - 1],
        &[x.clone(), y.clone()],
        4,
    );
}

#[test]
fn system_cubic_and_line_has_complex_pair() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    let sols = check_system(&[x.powi(3) - &y, &x + &y - 2], &[x.clone(), y.clone()], 3);
    let complex = sols
        .iter()
        .filter(|s| s[0].eval_complex64().unwrap().1.abs() > 1e-9)
        .count();
    assert_eq!(complex, 2);
}

#[test]
fn system_two_independent_quadratics() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    check_system(&[x.powi(2) - 2, y.powi(2) - 3], &[x.clone(), y.clone()], 4);
}

#[test]
fn system_symmetric_parabolas() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    check_system(
        &[x.powi(2) + &y - 1, y.powi(2) + &x - 1],
        &[x.clone(), y.clone()],
        4,
    );
}

#[test]
fn system_parabola_pair_with_complex_solutions() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    let sols = check_system(
        &[x.powi(2) - &y, y.powi(2) - &x],
        &[x.clone(), y.clone()],
        4,
    );
    let complex = sols
        .iter()
        .filter(|s| s[0].eval_complex64().unwrap().1.abs() > 1e-9)
        .count();
    assert_eq!(complex, 2, "(-1/2 ± i√3/2) pair expected");
}

#[test]
fn system_circle_and_parabola() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    check_system(
        &[x.powi(2) + y.powi(2) - 1, x.powi(2) - &y],
        &[x.clone(), y.clone()],
        4,
    );
}

#[test]
fn system_three_variables_elementary_symmetric() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    let z = ctx.symbol("z");
    check_system(
        &[
            &x + &y + &z - 6,
            &x * &y + &y * &z + &z * &x - 11,
            &x * &y * &z - 6,
        ],
        &[x.clone(), y.clone(), z.clone()],
        6,
    );
}

#[test]
fn system_three_variables_sphere_and_planes() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    let z = ctx.symbol("z");
    check_system(
        &[x.powi(2) + y.powi(2) + z.powi(2) - 3, &x - &y, &y - &z],
        &[x.clone(), y.clone(), z.clone()],
        2,
    );
}

#[test]
fn system_purely_complex_solutions() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    let sols = check_system(&[x.powi(2) + 1, &y - &x], &[x.clone(), y.clone()], 2);
    for s in &sols {
        assert!(s[0].eval_complex64().unwrap().1.abs() > 0.5);
    }
}

#[test]
fn system_quintic_eliminant_uses_rootof_and_verifies() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    let sols = check_system(
        &[x.powi(5) - &x - 1, &y - x.powi(2)],
        &[x.clone(), y.clone()],
        5,
    );
    assert!(
        sols.iter().any(|s| format!("{}", s[1]).contains("RootOf")),
        "quintic roots should be RootOf placeholders: {sols:?}"
    );
}

#[test]
fn system_inconsistent_returns_empty() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    let sols = symplex::polysys::solve_system_ex(
        &[x.powi(2) + y.powi(2) - 1, x.powi(2) + y.powi(2) - 4],
        &[x.clone(), y.clone()],
    )
    .unwrap();
    assert!(sols.is_empty());
}

#[test]
fn system_double_root_reported_once() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    check_system(
        &[x.powi(2) - 2 * &x + 1, &y - &x],
        &[x.clone(), y.clone()],
        1,
    );
}

#[test]
fn system_cube_root_of_two() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    check_system(&[x.powi(3) - 2, &y - x.powi(2)], &[x.clone(), y.clone()], 3);
}

#[test]
fn system_four_variables_triangular() {
    let ctx = Context::new();
    let (x, y) = xy(&ctx);
    let (z, w) = (ctx.symbol("z"), ctx.symbol("w"));
    check_system(
        &[w.powi(2) - 2, &z - &w - 1, y.powi(2) - &z, &x - &y * &w],
        &[x.clone(), y.clone(), z.clone(), w.clone()],
        4,
    );
}
