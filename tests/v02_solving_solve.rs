//! 0.2 solving campaign — univariate `solve` semantics, general (periodic)
//! solutions, absolute-value equations/inequalities, `check_solution`, and
//! the safeguarded numeric root finder.
//!
//! Every returned solution is verified by substitution back into the
//! equation (exact or numeric).

use symplex::prelude::*;

/// Assert that every root satisfies `expr = 0` (complex-aware numeric check).
fn verify_roots(expr: &Ex, var: &Ex, roots: &[Ex], label: &str) {
    assert!(!roots.is_empty(), "{label}: no roots returned");
    for r in roots {
        let residual = expr.subs(var, r).eval();
        if residual.is_zero_structural() {
            continue;
        }
        let (re, im) = residual
            .eval_complex64()
            .unwrap_or_else(|e| panic!("{label}: cannot evaluate residual for root {r}: {e}"));
        assert!(
            re.hypot(im) < 1e-9,
            "{label}: root {r} gives residual {re} + {im}i"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Identity / contradiction semantics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn identity_is_infinite_solutions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(matches!(
        ctx.int(0).solve(&x),
        Err(SymplexError::InfiniteSolutions { .. })
    ));
    // x - x canonicalises to 0.
    assert!(matches!(
        (&x - &x).solve(&x),
        Err(SymplexError::InfiniteSolutions { .. })
    ));
    // Independent of x and zero after evaluation: sin(0).
    assert!(matches!(
        ctx.int(0).sin().solve(&x),
        Err(SymplexError::InfiniteSolutions { .. })
    ));
    // (x+1)^2 - x^2 - 2x - 1 is polynomial zero.
    let e = &(&x + 1).powi(2) - &x.powi(2) - &x * 2 - 1;
    assert!(matches!(
        e.solve(&x),
        Err(SymplexError::InfiniteSolutions { .. })
    ));
}

#[test]
fn nonzero_constant_is_no_solution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for c in [1, -3, 7] {
        assert!(
            matches!(ctx.int(c).solve(&x), Err(SymplexError::NoSolution { .. })),
            "{c} = 0 should be NoSolution"
        );
    }
    // pi is a nonzero constant too.
    assert!(matches!(
        ctx.pi().solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
    // Symbolic constant independent of x: reported as NoSolution with a reason.
    let a = ctx.symbol("a");
    match a.solve(&x) {
        Err(SymplexError::NoSolution { reason, .. }) => {
            assert!(reason.contains("does not depend"), "{reason}")
        }
        other => panic!("expected NoSolution, got {other:?}"),
    }
}

#[test]
fn range_restrictions_are_no_solution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(matches!(
        x.exp().solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
    assert!(matches!(
        (&x.exp() + 1).solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
    assert!(matches!(
        (&x.sin() - 2).solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
    assert!(matches!(
        (&x.abs() + 1).solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
    // 1/x = 0 has only an infinite "candidate" — no solution.
    assert!(matches!(
        x.powi(-1).solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
}

#[test]
fn solve_or_empty_is_empty_for_degenerate_cases() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(ctx.int(0).solve_or_empty(&x).is_empty());
    assert!(ctx.int(1).solve_or_empty(&x).is_empty());
    assert_eq!((&x - 3).solve_or_empty(&x).len(), 1);
}

#[test]
fn solve_as_set_identity_and_contradiction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(format!("{}", ctx.int(0).solve_as_set(&x)), "UniversalSet");
    assert_eq!(format!("{}", ctx.int(1).solve_as_set(&x)), "EmptySet");
    assert_eq!(format!("{}", x.exp().solve_as_set(&x)), "EmptySet");
    let s = format!("{}", (&x.powi(2) - 4).solve_as_set(&x));
    assert!(s.contains('2') && s.contains("-2"), "{s}");
}

#[test]
fn equation_type_follows_new_semantics() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let identity = Equation::new(&x + 1, &x + 1);
    assert!(matches!(
        identity.solve(&x),
        Err(SymplexError::InfiniteSolutions { .. })
    ));
    let contradiction = Equation::new(&x + 1, &x + 2);
    assert!(matches!(
        contradiction.solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
}

#[test]
fn genuine_equations_still_solve() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: Vec<(Ex, usize)> = vec![
        (&x.powi(2) - &x * 5 + 6, 2),
        (&x.powi(2) + 1, 2),
        (&x.powi(3) - 8, 3),
        (&x.powi(3) - 2, 3),
        (&x.powi(4) + 4, 4),
        (&x.powi(5) - 2, 5),
        (&x.powi(6) - 1, 6),
        (&x.exp() - 5, 1),
        (&x.ln() - 2, 1),
        (&x.sqrt() - 3, 1),
        (&x.exp() * &x - 1, 1),
    ];
    for (e, n) in cases {
        let roots = e.solve(&x).unwrap_or_else(|err| panic!("{e}: {err}"));
        assert_eq!(roots.len(), n, "{e}: {roots:?}");
        verify_roots(&e, &x, &roots, &format!("{e}"));
    }
}

#[test]
fn binomial_roots_are_explicit_not_rootof() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.powi(5) - 2;
    let roots = e.solve(&x).unwrap();
    assert_eq!(roots.len(), 5);
    for r in &roots {
        assert!(!r.has_unevaluated(), "root should be explicit: {r}");
    }
    verify_roots(&e, &x, &roots, "x^5 - 2");
    // Distinct as complex numbers.
    let mut vals: Vec<(f64, f64)> = roots.iter().map(|r| r.eval_complex64().unwrap()).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for w in vals.windows(2) {
        assert!((w[0].0 - w[1].0).abs() + (w[0].1 - w[1].1).abs() > 1e-9);
    }
}

#[test]
fn symbolic_coefficient_quadratic() {
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    let e = &x.powi(2) - &a;
    let roots = e.solve(&x).unwrap();
    assert_eq!(roots.len(), 2);
    for r in &roots {
        let res = e.subs(&x, r).simplify();
        assert!(res.is_zero_structural(), "{r} → {res}");
    }
    // a x^2 + b x + 1 = 0 — general quadratic formula.
    let e = &(&a * &x.powi(2)) + &(&b * &x) + 1;
    let roots = e.solve(&x).unwrap();
    assert_eq!(roots.len(), 2);
    for r in &roots {
        // Check numerically at a = 2, b = 5.
        let res = e
            .subs(&x, r)
            .subs_i64(&a, 2)
            .subs_i64(&b, 5)
            .eval_f64()
            .unwrap();
        assert!(res.abs() < 1e-9, "{r} → {res}");
    }
}

#[test]
fn sin_of_linear_argument() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x * 2 + 1).sin() - &ctx.rational(1, 2);
    let roots = e.solve(&x).unwrap();
    assert_eq!(roots.len(), 2);
    verify_roots(&e, &x, &roots, "sin(2x+1) = 1/2");
}

// ═══════════════════════════════════════════════════════════════════════════
// General (periodic) solutions
// ═══════════════════════════════════════════════════════════════════════════

fn verify_family(eq: &Ex, x: &Ex, fam: &symplex::polysys::GeneralSolution, label: &str) {
    assert!(!fam.solutions.is_empty(), "{label}: empty");
    for k in -3..=3 {
        for s in fam.instance(k) {
            let (re, im) = eq.subs(x, &s).eval_complex64().unwrap();
            assert!(
                re.hypot(im) < 1e-9,
                "{label}: n={k}, x={s}: residual {re}+{im}i"
            );
        }
    }
}

#[test]
fn general_sin_equals_half() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x.sin() - &ctx.rational(1, 2);
    let fam = eq.solve_general(&x).unwrap();
    assert_eq!(fam.solutions.len(), 2);
    assert_eq!(fam.parameters.len(), 1);
    assert_eq!(format!("{}", fam.parameters[0]), "n");
    assert_eq!(fam.parameters[0].is_integer(), Some(true));
    for s in &fam.solutions {
        assert!(s.contains(&fam.parameters[0]), "family should use n: {s}");
        assert!(format!("{s}").contains("pi"), "{s}");
    }
    verify_family(&eq, &x, &fam, "sin x = 1/2");
    // Instances at n = 0 are the principal solutions π/6 and 5π/6.
    let inst = fam.instance(0);
    let mut vals: Vec<f64> = inst.iter().map(|e| e.eval_f64().unwrap()).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((vals[0] - std::f64::consts::PI / 6.0).abs() < 1e-12);
    assert!((vals[1] - 5.0 * std::f64::consts::PI / 6.0).abs() < 1e-12);
}

#[test]
fn general_cos_and_tan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x.cos() - &ctx.rational(1, 2);
    let fam = eq.solve_general(&x).unwrap();
    assert_eq!(fam.solutions.len(), 2);
    verify_family(&eq, &x, &fam, "cos x = 1/2");
    let eq = &x.tan() - 1;
    let fam = eq.solve_general(&x).unwrap();
    assert_eq!(fam.solutions.len(), 1);
    assert!(format!("{}", fam.solutions[0]).contains("pi"));
    verify_family(&eq, &x, &fam, "tan x = 1");
}

#[test]
fn general_linear_argument_and_change_of_variable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(3x - 1) = 1/2
    let eq = &(&x * 3 - 1).sin() - &ctx.rational(1, 2);
    let fam = eq.solve_general(&x).unwrap();
    assert_eq!(fam.solutions.len(), 2);
    verify_family(&eq, &x, &fam, "sin(3x-1) = 1/2");
    // cos²x = 1/4  →  four families
    let eq = &x.cos().powi(2) - &ctx.rational(1, 4);
    let fam = eq.solve_general(&x).unwrap();
    assert_eq!(fam.solutions.len(), 4);
    verify_family(&eq, &x, &fam, "cos^2 x = 1/4");
    // sin²x - sin x = 0
    let eq = &x.sin().powi(2) - &x.sin();
    let fam = eq.solve_general(&x).unwrap();
    assert!(fam.solutions.len() >= 3, "{:?}", fam.solutions);
    verify_family(&eq, &x, &fam, "sin^2 - sin = 0");
}

#[test]
fn general_parameter_names_are_fresh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let n = ctx.symbol("n");
    let eq = &x.sin() - &ctx.rational(1, 2);
    let fam = eq.solve_general(&x).unwrap();
    assert_ne!(fam.parameters[0], n, "must not reuse the user's n");
    assert_eq!(format!("{}", fam.parameters[0]), "n1");
    let gen2 = eq.solve_general(&x).unwrap();
    assert_eq!(format!("{}", gen2.parameters[0]), "n2");
}

#[test]
fn general_polynomial_has_no_parameters() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x.powi(2) - 4;
    let fam = eq.solve_general(&x).unwrap();
    assert_eq!(fam.solutions.len(), 2);
    assert!(fam.parameters.is_empty());
    assert!(matches!(
        ctx.int(0).solve_general(&x),
        Err(SymplexError::InfiniteSolutions { .. })
    ));
    assert!(matches!(
        ctx.int(2).solve_general(&x),
        Err(SymplexError::NoSolution { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// Absolute value
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn abs_equation_two_branches() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x - 2).abs() - 3;
    let roots = e.solve(&x).unwrap();
    assert_eq!(roots.len(), 2);
    verify_roots(&e, &x, &roots, "|x-2| = 3");
    let mut v: Vec<f64> = roots.iter().map(|r| r.eval_f64().unwrap()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(v, vec![-1.0, 5.0]);
    // |2x + 1| = 0 → single root -1/2
    let e = (&x * 2 + 1).abs();
    let roots = e.solve(&x).unwrap();
    assert_eq!(roots.len(), 1);
    assert_eq!(format!("{}", roots[0]), "-1/2");
}

#[test]
fn abs_inequalities() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x - 2).abs() - 3;
    assert_eq!(format!("{}", e.solve_lt(&x)), "(-1, 5)");
    assert_eq!(format!("{}", e.solve_le(&x)), "[-1, 5]");
    let gt = format!("{}", e.solve_gt(&x));
    assert!(gt.contains("(-oo, -1)") && gt.contains("(5, oo)"), "{gt}");
    let ge = format!("{}", e.solve_ge(&x));
    assert!(ge.contains("-1]") && ge.contains("[5"), "{ge}");
    // Symbolic centre: |x - a| <= 3 → [a - 3, a + 3]
    let a = ctx.symbol("a");
    let e = &(&x - &a).abs() - 3;
    assert_eq!(format!("{}", e.solve_le(&x)), "[a - 3, a + 3]");
    // Negative coefficient flips the relation: -|x| + 1 > 0 ⇔ |x| < 1
    let e = &ctx.int(1) - &x.abs();
    assert_eq!(format!("{}", e.solve_gt(&x)), "(-1, 1)");
    // |x| + 1 < 0 impossible, 2|x| + 1 > 0 always
    assert_eq!(format!("{}", (&x.abs() + 1).solve_lt(&x)), "EmptySet");
    assert_eq!(format!("{}", (&x.abs() * 2 + 1).solve_gt(&x)), "(-oo, oo)");
}

// ═══════════════════════════════════════════════════════════════════════════
// check_solution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn check_solution_definite_false() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = &x.powi(2) - 4;
    assert_eq!(p.check_solution(&x, &ctx.int(2)), Some(true));
    assert_eq!(p.check_solution(&x, &ctx.int(3)), Some(false));
    // Complex residual: x^2 + 1 at 2i gives -3 → false; at i → true.
    let q = &x.powi(2) + 1;
    assert_eq!(q.check_solution(&x, &ctx.i_unit()), Some(true));
    assert_eq!(q.check_solution(&x, &(&ctx.i_unit() * 2)), Some(false));
    // Provably nonzero symbolic residual via assumptions.
    let p_sym = ctx.symbol_with("p", &[Assumption::Positive]);
    let r = &x - &p_sym;
    assert_eq!(r.check_solution(&x, &ctx.int(0)), Some(false));
    // Irrational exact root.
    let s = &x.powi(2) - 2;
    assert_eq!(s.check_solution(&x, &ctx.int(2).sqrt()), Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric root finding
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_numeric_safeguards() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Classic divergence of undamped Newton for atan from |x0| > 1.39.
    let r = x.atan().solve_numeric(&x, 3.0, 100, 1e-12).unwrap();
    assert!(r.abs() < 1e-8);
    // Starting where f' = 0 (x = 0 for x^2 - 2): secant/perturbation kicks in.
    let r = (&x.powi(2) - 2).solve_numeric(&x, 0.0, 100, 1e-12).unwrap();
    assert!((r.abs() - std::f64::consts::SQRT_2).abs() < 1e-9);
    // Ordinary case.
    let r = (&x - &x.cos()).solve_numeric(&x, 1.0, 50, 1e-12).unwrap();
    assert!((r - 0.739_085_133_215_160_6).abs() < 1e-10);
    // Non-convergence reports the residual.
    let err = (&x.powi(2) + 1)
        .solve_numeric(&x, 1.0, 20, 1e-12)
        .unwrap_err();
    assert!(format!("{err}").contains("residual"), "{err}");
}
