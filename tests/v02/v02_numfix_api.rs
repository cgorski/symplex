//! Regression tests for small API-level numeric bugs fixed in 0.2 numfix:
//!
//! * `Context::rational(p, 0)` panicked inside `num-rational`.
//! * `abs(3 + 4i)` was not folded to `5` by `eval`/`simplify`.
//! * `expr_type()` reported `Unevaluated` for `RootOf` while
//!   `has_unevaluated()` (correctly) did not.

use symplex::expr::ExprType;
use symplex::prelude::*;

// ── expr_type / has_unevaluated consistency for RootOf ───────────────────────

#[test]
fn rootof_is_a_constant_not_unevaluated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁵ − x − 1 is irreducible over ℚ: the solver returns RootOf values.
    let roots = (&x.powi(5) - &x - 1).solve_or_empty(&x);
    assert_eq!(roots.len(), 5);
    for r in &roots {
        assert!(format!("{r}").contains("RootOf"), "{r}");
        assert!(!r.has_unevaluated(), "{r} is a complete algebraic value");
        assert_ne!(r.expr_type(), ExprType::Unevaluated, "{r}");
        assert_eq!(r.expr_type(), ExprType::Constant, "{r}");
        assert!(r.free_symbols().is_empty(), "{r}");
    }
}

#[test]
fn rootof_inside_expression_is_not_unevaluated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x.powi(5) - &x - 1).solve_or_empty(&x).remove(0);
    let e = &r * 2 + 1;
    assert!(!e.has_unevaluated());
    assert_eq!(e.expr_type(), ExprType::Add);
    // Genuinely formal nodes still report both ways.
    let lim = x.sin().limit(&x, &ctx.int(0));
    if lim.has_unevaluated() {
        assert_eq!(lim.expr_type(), ExprType::Unevaluated);
    }
    let integ = x.exp().pow(&x.powi(2)).integrate(&x);
    if integ.has_unevaluated() {
        assert_eq!(integ.expr_type(), ExprType::Integral);
    }
}

// ── abs of numeric complex constants ─────────────────────────────────────

#[test]
fn abs_of_3_plus_4i_is_5() {
    let ctx = Context::new();
    let z = ctx.int(3) + ctx.int(4) * ctx.i_unit();
    assert_eq!(z.abs().eval(), ctx.int(5));
    assert_eq!(z.abs().simplify(), ctx.int(5));
    // abs_squared already worked; keep them consistent.
    assert_eq!(z.abs_squared().eval(), ctx.int(25));
}

#[test]
fn abs_of_pure_imaginary_units() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(i.abs().eval(), ctx.int(1), "abs(i)");
    assert_eq!((ctx.int(2) * &i).abs().eval(), ctx.int(2), "abs(2i)");
    assert_eq!((ctx.int(-2) * &i).abs().eval(), ctx.int(2), "abs(-2i)");
    assert_eq!((-&i).abs().eval(), ctx.int(1), "abs(-i)");
}

#[test]
fn abs_of_complex_constant_with_irrational_modulus() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(
        (ctx.int(1) + &i).abs().eval(),
        ctx.int(2).sqrt(),
        "abs(1+i)"
    );
    assert_eq!(
        (ctx.rational(1, 2) + ctx.rational(1, 3) * &i).abs().eval(),
        ctx.rational(1, 6) * ctx.int(13).sqrt(),
        "abs(1/2 + i/3)"
    );
}

#[test]
fn abs_of_complex_constant_with_radical_parts() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    // |1 + √3 i| = 2
    assert_eq!(
        (ctx.int(1) + ctx.int(3).sqrt() * &i).abs().eval(),
        ctx.int(2)
    );
    // |(1 + i)²| = |2i| = 2
    assert_eq!((ctx.int(1) + &i).powi(2).abs().eval(), ctx.int(2));
    // |e + π i| = √(e² + π²)
    let z = (ctx.e() + ctx.pi() * &i).abs().eval();
    let want = (2.0f64.exp() + std::f64::consts::PI.powi(2)).sqrt();
    assert!(!format!("{z}").contains("abs"), "{z}");
    assert!((z.eval_f64().unwrap() - want).abs() < 1e-14, "{z}");
}

#[test]
fn abs_fold_leaves_symbolic_and_real_arguments_alone() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let x = ctx.symbol("x");
    let z = (&x + &i).abs();
    assert_eq!(z.eval(), z, "abs(x + i) must stay symbolic");
    // A real constant is not rewritten as √(π²).
    assert_eq!(format!("{}", ctx.pi().abs().eval()), "abs(pi)");
    // Transcendental parts are not expanded into √(sin² + cos²).
    assert_eq!(format!("{}", i.exp().abs().eval()), "abs(exp(I))");
}

// ── Context::rational with zero denominator ────────────────────────────

#[test]
fn rational_with_zero_denominator_is_complex_infinity() {
    let ctx = Context::new();
    assert_eq!(ctx.rational(1, 0), ctx.complex_infinity());
    assert_eq!(ctx.rational(-7, 0), ctx.complex_infinity());
    assert_eq!(ctx.rational(i64::MAX, 0), ctx.complex_infinity());
    // Same value as division builds.
    assert_eq!(ctx.rational(5, 0), ctx.int(5) / ctx.int(0));
    assert_eq!(
        format!("{}", ctx.rational(1, 0)),
        format!("{}", ctx.complex_infinity())
    );
}

#[test]
fn rational_zero_over_zero_is_nan() {
    let ctx = Context::new();
    assert_eq!(ctx.rational(0, 0), ctx.nan());
    assert_eq!(ctx.rational(0, 0), ctx.int(0) / ctx.int(0));
}

#[test]
fn rational_nonzero_denominator_unchanged() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.rational(6, -4)), "-3/2");
    assert_eq!(ctx.rational(4, 2), ctx.int(2));
    assert_eq!(ctx.rational(0, 5), ctx.int(0));
}
