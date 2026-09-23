//! 0.24 — integration defects found by running the Rubi test suite
//! (`rubi-harness/`, 72,254 integrands judged by differentiation) and by
//! `fuzz_integrate`.  Each antiderivative is checked by `F′ = f` at points
//! on both sides of every gap in the real domain; the integrands have no
//! free parameters, or fixed rational ones.  A symplex antiderivative is a
//! real-variable one: `F′ = f` must hold wherever `f` is real.

use symplex::prelude::*;

/// `F = ∫ f` is a closed form with `F′ = f` at every `x` in `points`
/// (after substituting `params`).
fn antiderivative_at(f: &Ex, x: &Ex, params: &[(&Ex, &Ex)], points: &[Ex]) -> Ex {
    let big_f = f.integrate(x);
    assert!(!big_f.has_unevaluated(), "∫ {f} dx is unevaluated: {big_f}");
    let df = big_f.diff(x).subs_map(params);
    let ff = f.subs_map(params);
    for v in points {
        let (a, b) = (
            df.subs(x, v).eval_complex64().unwrap(),
            ff.subs(x, v).eval_complex64().unwrap(),
        );
        assert!(
            (a - b).norm() < 1e-10 * b.norm().max(1.0),
            "∫ {f} dx = {big_f}: F′ = {a} but f = {b} at x = {v}"
        );
    }
    big_f
}

/// `∫ f` is unevaluated, or a closed form with `F′ = f` at `points`.
fn never_wrong(f: &Ex, x: &Ex, points: &[Ex]) {
    let big_f = f.integrate(x);
    if big_f.has_unevaluated() {
        return;
    }
    let df = big_f.diff(x);
    for v in points {
        let (a, b) = (
            df.subs(x, v).eval_complex64().unwrap(),
            f.subs(x, v).eval_complex64().unwrap(),
        );
        assert!(
            (a - b).norm() < 1e-10 * b.norm().max(1.0),
            "∫ {f} dx = {big_f}: F′ = {a} but f = {b} at x = {v}"
        );
    }
}

#[test]
fn derivative_of_acosh_is_the_principal_one_below_minus_one() {
    // SymPy: diff(acosh(x), x) = 1/(sqrt(x - 1)*sqrt(x + 1));
    // mpmath: diff(acosh, -2) = -0.577350269189625764509148780502
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d = x.acosh().diff(&x).subs_i64(&x, -2).eval_f64().unwrap();
    assert!((d - -0.577_350_269_189_625_8).abs() < 1e-15, "{d}");
}

#[test]
fn acosh_and_the_square_root_forms_hold_on_both_sides_of_the_gap() {
    // acosh(x/a) as the antiderivative of 1/√(x²−a²) is right only for x > a.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts = [ctx.rational(-7, 2), ctx.rational(7, 2), ctx.int(-5)];
    antiderivative_at(&x.acosh(), &x, &[], &pts);
    antiderivative_at(&(&x * 2 + 1).acosh(), &x, &[], &pts);
    antiderivative_at(&(x.powi(2) - 4).sqrt(), &x, &[], &pts);
    antiderivative_at(&(1 / (x.powi(2) - 4).sqrt()), &x, &[], &pts);
    antiderivative_at(&(1 / (x.powi(2) + &x * 2 - 3).sqrt()), &x, &[], &pts);
    // Rubi (Hearn 202): 10/√(x²−4) + 1/√(x²−1)
    let f = 10 / (x.powi(2) - 4).sqrt() + 1 / (x.powi(2) - 1).sqrt();
    antiderivative_at(&f, &x, &[], &pts);
}

#[test]
fn a_symbolic_coefficient_is_not_a_zero_coefficient() {
    // 0.23: ∫ (a x + b)/(x² + 1) dx = 0 — a coefficient without a numeric
    // value was dropped as if it were zero (56 Rubi entries).
    // SymPy: integrate((b + a*x)/(1 + x**2), x) = (a/2 - I*b/2)*log(x - I) + (a/2 + I*b/2)*log(x + I)
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    let params = [(&a, &ctx.rational(6, 5)), (&b, &ctx.rational(3, 4))];
    let pts = [ctx.rational(1, 3), ctx.rational(-13, 4)];
    let cases = [
        (&b + &a * &x) / (x.powi(2) + 1),
        &x / (&a + &b * x.powi(2)),
        &x / (&a - &b * x.powi(2)),
        (&a + &x) / (a.powi(2) + x.powi(2)),
    ];
    for f in &cases {
        let big_f = antiderivative_at(f, &x, &params, &pts);
        assert!(!big_f.is_zero_structural(), "∫ {f} dx = 0");
    }
}

#[test]
fn square_root_of_a_perfect_square_keeps_its_sign() {
    // √((3x − 2)²) = 3|x − 2/3|: ∫ dx/√(9x² − 12x + 4) = sign(x − 2/3)·ln|x − 2/3|/3.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let q = ctx.int(9) * x.powi(2) - &x * 12 + 4;
    let pts = [ctx.rational(1, 3), ctx.rational(7, 5), ctx.rational(-13, 4)];
    antiderivative_at(&(1 / q.sqrt()), &x, &[], &pts);
    antiderivative_at(&(&x / q.sqrt()), &x, &[], &pts);
}

#[test]
fn logarithm_of_a_complex_argument_is_not_ln_abs() {
    // Rubi 6.1.5 e165: ∫ cosh x/(i + sinh x) dx = ln(i + sinh x), not ln|i + sinh x|
    // (whose derivative is the conjugate).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.i_unit();
    let pts = [ctx.rational(1, 3), ctx.rational(-5, 7)];
    antiderivative_at(&(x.cosh() / (&i + x.sinh())), &x, &[], &pts);
    // Rubi 4.4.1.2 e6: csc²x/(i + cot x)
    antiderivative_at(
        &(x.sin().powi(-2) / (&i + x.cos() / x.sin())),
        &x,
        &[],
        &pts,
    );
}

#[test]
fn abs_of_a_non_real_base_is_not_squared_away() {
    // fuzz_integrate: ∫ |√x|² dx was x²/2; |√x|² = |x|, which is −x for x < 0.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts = [ctx.rational(-5, 7), ctx.rational(7, 5)];
    never_wrong(&x.sqrt().abs().powi(2), &x, &pts);
    never_wrong(&(&x - 1).sqrt().abs().powi(2), &x, &[ctx.rational(1, 3)]);
    never_wrong(&x.ln().abs().powi(2), &x, &pts);
    // For a real base the rewrite still applies: ∫ |x|² dx = x³/3.
    antiderivative_at(&x.abs().powi(2), &x, &[], &pts);
}
