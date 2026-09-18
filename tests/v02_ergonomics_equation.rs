//! v0.2 ergonomics — the `Equation` type.

use symplex::eq::Equation;
use symplex::prelude::*;

fn setup() -> (Context, Ex) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    (ctx, x)
}

#[test]
fn accessors_and_display() {
    let (ctx, x) = setup();
    let eq = Equation::new(&x + 1, ctx.int(5));
    assert_eq!(eq.lhs(), &(&x + 1));
    assert_eq!(eq.rhs(), &ctx.int(5));
    assert_eq!(format!("{eq}"), "x + 1 = 5");
    assert_eq!(format!("{eq:?}"), "Equation(x + 1 = 5)");
}

#[test]
fn swap_is_an_involution() {
    let (ctx, x) = setup();
    let eq = Equation::new(x.clone(), ctx.int(2));
    assert_eq!(format!("{}", eq.swap()), "2 = x");
    assert_eq!(eq.swap().swap(), eq);
}

#[test]
fn partial_eq_structural() {
    let (ctx, x) = setup();
    assert_eq!(
        Equation::new(&x + 1, ctx.int(5)),
        Equation::new(1 + &x, ctx.int(5))
    );
    assert_ne!(
        Equation::new(&x + 1, ctx.int(5)),
        Equation::new(x.clone(), ctx.int(4))
    );
}

#[test]
fn rearrange_linear_equation_step_by_step() {
    let (ctx, x) = setup();
    let eq = Equation::new(&x * 3 - 6, ctx.int(9)); // 3x - 6 = 9
    let eq = &eq + 6; // 3x = 15
    assert_eq!(eq, Equation::new(&x * 3, ctx.int(15)));
    let eq = eq / 3; // x = 5
    assert_eq!(eq, Equation::new(x.clone(), ctx.int(5)));
}

#[test]
fn arithmetic_with_every_scalar_kind() {
    let (ctx, x) = setup();
    let eq = Equation::new(x.clone(), ctx.int(1));
    assert_eq!(&eq + 1, Equation::new(&x + 1, ctx.int(2)));
    assert_eq!(&eq + 1_i64, Equation::new(&x + 1, ctx.int(2)));
    assert_eq!(&eq * 0.5, Equation::new(&x / 2, ctx.rational(1, 2)));
    assert_eq!(
        &eq - num_bigint::BigInt::from(3),
        Equation::new(&x - 3, ctx.int(-2))
    );
    assert_eq!(&eq * &x, Equation::new(x.powi(2), x.clone()));
    assert_eq!(&eq - x.clone(), Equation::new(ctx.zero(), 1 - &x));
}

#[test]
fn equation_with_equation_combines_sides() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e1 = Equation::new(&x + &y, ctx.int(10));
    let e2 = Equation::new(&x - &y, ctx.int(4));
    assert_eq!(&e1 + &e2, Equation::new(&x * 2, ctx.int(14)));
    assert_eq!(&e1 - &e2, Equation::new(&y * 2, ctx.int(6)));
    assert_eq!(
        e1.clone() * e2.clone(),
        Equation::new(&(&x + &y) * &(&x - &y), ctx.int(40))
    );
    assert_eq!(
        e1 / e2,
        Equation::new(&(&x + &y) / &(&x - &y), ctx.rational(5, 2))
    );
}

#[test]
fn negation_flips_both_sides() {
    let (ctx, x) = setup();
    let eq = Equation::new(&x - 1, ctx.int(3));
    assert_eq!(-&eq, Equation::new(1 - &x, ctx.int(-3)));
    assert_eq!(-eq.clone(), -&eq);
}

#[test]
fn apply_function_to_both_sides() {
    let (ctx, x) = setup();
    let eq = Equation::new(x.exp(), ctx.int(5))
        .apply(|s| s.ln())
        .simplify();
    assert_eq!(eq.lhs(), &x);
    assert_eq!(eq.rhs(), &ctx.int(5).ln());
}

#[test]
fn to_zero_form() {
    let (ctx, x) = setup();
    let z = Equation::new(x.powi(2), &x + 6).to_zero_form();
    assert_eq!(z.rhs(), &ctx.zero());
    assert_eq!(z.lhs(), &(&x.powi(2) - &x - 6));
    assert_eq!(z.to_expr(), z.lhs().clone());
}

#[test]
fn solve_and_solve_for() {
    let (ctx, x) = setup();
    let eq = Equation::new(x.powi(2), &x + 6);
    let mut roots = eq.solve(&x).unwrap();
    roots.sort_by(|a, b| a.compare_numeric(b).unwrap());
    assert_eq!(roots, vec![ctx.int(-2), ctx.int(3)]);

    let sols = eq.solve_for(&x).unwrap();
    assert_eq!(sols.len(), 2);
    for s in &sols {
        assert_eq!(s.lhs(), &x);
        // Each solution satisfies the original equation.
        assert_eq!(eq.subs(&x, s.rhs()).eval().is_identity(), Some(true));
    }
}

#[test]
fn solve_for_no_solution_is_an_error() {
    let (ctx, x) = setup();
    // x = x + 1 is inconsistent.
    let res = Equation::new(x.clone(), &x + 1).solve_for(&x);
    assert!(res.is_err() || res.unwrap().is_empty());
    let _ = ctx;
}

#[test]
fn is_identity_and_is_satisfied() {
    let (ctx, x) = setup();
    let ident = Equation::new((&x + 1).powi(2), &x.powi(2) + &x * 2 + 1);
    assert_eq!(ident.is_identity(), Some(true));
    assert_eq!(ident.is_satisfied(), Some(true));
    assert_eq!(Equation::new(x.clone(), &x + 1).is_identity(), Some(false));
    assert_eq!(Equation::new(x.clone(), ctx.int(2)).is_identity(), None);
    assert_eq!(
        Equation::new(x.clone(), ctx.int(2))
            .subs_i64(&x, 2)
            .is_satisfied(),
        Some(true)
    );
}

#[test]
fn simplify_expand_eval_both_sides() {
    let (ctx, x) = setup();
    let eq = Equation::new((&x + 1).powi(2), &ctx.int(2).sqrt() * &ctx.int(2).sqrt());
    let ex = eq.expand();
    assert_eq!(ex.lhs(), &(&x.powi(2) + &x * 2 + 1));
    assert_eq!(eq.eval().rhs(), &ctx.int(2));
    let trig = Equation::new(&x.sin().powi(2) + &x.cos().powi(2), ctx.one()).simplify();
    assert_eq!(trig.lhs(), &ctx.one());
}

#[test]
fn eq_macro_still_builds_equations() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = symplex::eq!(ctx, x + 1 = 5);
    assert_eq!(eq.lhs(), &(&x + 1));
    assert_eq!((&eq - 1).rhs(), &ctx.int(4));
}

#[test]
#[should_panic(expected = "different contexts")]
fn mixed_context_sides_panic() {
    let a = Context::new();
    let b = Context::new();
    let _ = Equation::new(a.symbol("x"), b.int(1));
}
