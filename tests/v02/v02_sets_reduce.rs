//! symplex 0.2 — `reduce_inequalities`, `BoolEx::solve_for`,
//! `SetEx::to_condition` round trips.

use symplex::prelude::*;

fn s<T: std::fmt::Display>(e: &T) -> String {
    format!("{e}")
}

#[test]
fn quadratic_and_linear_conjunction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let c1 = x.powi(2).gt(&ctx.int(4));
    let c2 = x.lt(&ctx.int(5));
    let sol = SetEx::reduce_inequalities(&[c1.clone(), c2.clone()], &x).unwrap();
    assert_eq!(s(&sol), "(-oo, -2) ∪ (2, 5)");
    // same thing as a single conjunction
    assert_eq!(c1.and(&c2).solve_for(&x).unwrap(), sol);
    // membership sanity
    assert_eq!(sol.contains(&ctx.int(3)), Some(true));
    assert_eq!(sol.contains(&ctx.int(-3)), Some(true));
    assert_eq!(sol.contains(&ctx.int(0)), Some(false));
    assert_eq!(sol.contains(&ctx.int(5)), Some(false));
    assert_eq!(sol.contains(&ctx.int(2)), Some(false));
}

#[test]
fn disjunction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.ge(&ctx.int(0)).or(&x.lt(&ctx.int(-3)));
    assert_eq!(s(&cond.solve_for(&x).unwrap()), "(-oo, -3) ∪ [0, oo)");
}

#[test]
fn negation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = x.powi(2).le(&ctx.int(1)).not();
    assert_eq!(s(&cond.solve_for(&x).unwrap()), "(-oo, -1) ∪ (1, oo)");
    // ¬(x > 0) = x ≤ 0
    assert_eq!(
        s(&x.gt(&ctx.int(0)).not().solve_for(&x).unwrap()),
        "(-oo, 0]"
    );
}

#[test]
fn equality_and_inequality_atoms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        s(&x.powi(2).eq_expr(&ctx.int(4)).solve_for(&x).unwrap()),
        "{-2, 2}"
    );
    assert_eq!(
        s(&x.ne_expr(&ctx.int(2)).solve_for(&x).unwrap()),
        "(-oo, 2) ∪ (2, oo)"
    );
    // no real roots
    assert_eq!(
        s(&(&x.powi(2) + 1).eq_expr(&ctx.int(0)).solve_for(&x).unwrap()),
        "EmptySet"
    );
    assert_eq!(
        s(&(&x.powi(2) + 1).ne_expr(&ctx.int(0)).solve_for(&x).unwrap()),
        "(-oo, oo)"
    );
    // x² ≥ 0 everywhere; x² < 0 nowhere; x² ≤ 0 only at 0
    assert_eq!(
        s(&x.powi(2).ge(&ctx.int(0)).solve_for(&x).unwrap()),
        "(-oo, oo)"
    );
    assert_eq!(
        s(&x.powi(2).lt(&ctx.int(0)).solve_for(&x).unwrap()),
        "EmptySet"
    );
    assert_eq!(s(&x.powi(2).le(&ctx.int(0)).solve_for(&x).unwrap()), "{0}");
}

#[test]
fn constants_and_true_false() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let always = ctx.int(1).gt(&ctx.int(0));
    let never = ctx.int(0).gt(&ctx.int(1));
    assert_eq!(
        s(&always.and(&x.gt(&ctx.int(0))).solve_for(&x).unwrap()),
        "(0, oo)"
    );
    assert_eq!(
        s(&never.or(&x.gt(&ctx.int(0))).solve_for(&x).unwrap()),
        "(0, oo)"
    );
    assert_eq!(
        s(&never.and(&x.gt(&ctx.int(0))).solve_for(&x).unwrap()),
        "EmptySet"
    );
    assert_eq!(s(&always.solve_for(&x).unwrap()), "UniversalSet");
}

#[test]
fn nested_boolean_structure() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x > 0 ∧ x < 3) ∨ (x ≥ 5 ∧ ¬(x > 7))  → (0, 3) ∪ [5, 7]
    let left = x.gt(&ctx.int(0)).and(&x.lt(&ctx.int(3)));
    let right = x.ge(&ctx.int(5)).and(&x.gt(&ctx.int(7)).not());
    assert_eq!(
        s(&left.or(&right).solve_for(&x).unwrap()),
        "(0, 3) ∪ [5, 7]"
    );
    // xor-style: exactly one of x > 0, x > 2  → (0, 2]
    let p = x.gt(&ctx.int(0));
    let q = x.gt(&ctx.int(2));
    assert_eq!(s(&p.xor(&q).solve_for(&x).unwrap()), "(0, 2]");
}

#[test]
fn irrational_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sol = x.powi(2).lt(&ctx.int(2)).solve_for(&x).unwrap();
    let d = s(&sol);
    assert!(
        d.starts_with('(') && d.ends_with(')') && d.contains("sqrt(2)"),
        "{d}"
    );
    assert_eq!(sol.contains(&ctx.int(1)), Some(true));
    assert_eq!(sol.contains(&ctx.rational(3, 2)), Some(false));
    assert_eq!(sol.contains(&ctx.rational(-7, 5)), Some(true));
    // combine with a rational window across the irrational endpoint
    let window = ctx.interval(&ctx.int(1), &ctx.int(2), false, false);
    assert_eq!(s(&sol.intersection(&window).simplify()), "[1, sqrt(2))");
}

#[test]
fn errors() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    assert!(SetEx::reduce_inequalities(&[], &x).is_err());
    assert!(
        y.gt(&ctx.int(0)).solve_for(&x).is_err(),
        "does not involve x"
    );
    assert!(
        x.gt(&ctx.int(0)).solve_for(&ctx.int(1)).is_err(),
        "not a symbol"
    );
    // opaque atom (a symbol used as a proposition)
    let p: BoolEx = ctx.symbol("p").gt(&ctx.int(0)).and(&x.gt(&ctx.int(0)));
    assert!(p.solve_for(&x).is_err());
}

#[test]
fn to_condition_roundtrip() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let set = ctx
        .interval(&ctx.int(0), &ctx.int(1), true, false)
        .union(&ctx.finite_set(&[ctx.int(3)]))
        .union(&ctx.interval(&ctx.int(5), &ctx.infinity(), false, true));
    let cond = set.to_condition(&x).unwrap();
    let d = s(&cond);
    assert!(
        d.contains("x > 0") && d.contains("1 >= x") && d.contains("x == 3") && d.contains("x >= 5"),
        "{d}"
    );
    assert_eq!(cond.solve_for(&x).unwrap(), set.simplify());
    // whole line and empty set
    assert_eq!(s(&ctx.reals().to_condition(&x).unwrap()), "True");
    assert_eq!(s(&ctx.empty_set().to_condition(&x).unwrap()), "False");
    // symbolic set → symbolic condition
    let y = ctx.symbol("y");
    let sym = ctx.interval(&y, &(&y + 1), false, true);
    assert_eq!(s(&sym.to_condition(&x).unwrap()), "x >= y & y + 1 > x");
    // complement → and-not
    let c = ctx.reals().complement(&ctx.finite_set(&[ctx.int(0)]));
    assert_eq!(s(&c.to_condition(&x).unwrap()), "!(x == 0)");
}

#[test]
fn solve_for_agrees_with_pointwise_evaluation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let conds: Vec<BoolEx> = vec![
        (&x.powi(3) - &x).gt(&ctx.int(0)),
        (&x.powi(2) - 3 * &x + 2)
            .le(&ctx.int(0))
            .or(&x.gt(&ctx.int(4))),
        (&x.powi(2) - 1)
            .ne_expr(&ctx.int(0))
            .and(&x.abs().lt(&ctx.int(3))),
        x.gt(&ctx.int(-1)).and(&x.lt(&ctx.int(1))).not(),
        (&x * 2 + 1)
            .ge(&ctx.int(0))
            .and(&(&x * -3).gt(&ctx.int(-9))),
    ];
    let points: Vec<Ex> = (-12..=12).map(|k| ctx.rational(k, 2)).collect();
    for cond in &conds {
        let sol = match cond.solve_for(&x) {
            Ok(sol) => sol,
            Err(e) => panic!("could not reduce {cond}: {e}"),
        };
        assert!(
            sol.as_intervals().is_some(),
            "{cond} → {sol} not in normal form"
        );
        for p in &points {
            let truth = s(&cond.subs(&x, p).eval());
            let expected = match truth.as_str() {
                "True" => true,
                "False" => false,
                other => panic!("{cond} at {p} did not evaluate: {other}"),
            };
            assert_eq!(sol.contains(p), Some(expected), "{cond} → {sol} at {p}");
        }
    }
}
