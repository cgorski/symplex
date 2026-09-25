//! After 0.28.0 — substitution into the variable of a derivative or an
//! indefinite integral, `RootOf` records its variable, and the `Matrix` /
//! `ExactMatrix` constructors return `Result`.
//!
//! - `f(x).diff(x).subs(x, 0)` was `Derivative(f(0), 0)` and
//!   `Integral(x²y, x).subs(x, 1/3)` was `Integral(1/9*y, 1/3)`: the
//!   variable of a `Derivative` / indefinite `Integral` was rewritten like
//!   any operand.  A value at a point is now the new `Subs` node (SymPy's
//!   `Subs`), or the derivative itself where `diff` can take it.
//! - `diff` built derivatives in non-symbols (`f(x²)′ = 2x·Derivative(f(x²),
//!   x²)`) and counted `f(x, x)′` twice; partial derivatives of an undefined
//!   function in a slot that is not a lone symbol are now `Subs` forms.
//! - A `RootOf` whose polynomial had a parameter recorded no variable, bound
//!   nothing, and `.subs(x, 1/3)` changed its meaning.
//! - `subs_algebraic` ignored binders; `try_replace` accepted a non-symbol
//!   bound variable; the dependence tests of summation, the Laplace and
//!   Z transforms and the ODE solver were structural.
//!
//! Reference values cite the SymPy 1.14 call they come from.

use symplex::prelude::*;

fn f_of(ctx: &Context, args: &[&Ex]) -> Ex {
    ctx.apply("f", args).unwrap()
}

// ── Part A: substitution into a variable slot ───────────────────────────

#[test]
fn a_derivative_at_a_point_is_a_subs_node() {
    // Up to 0.28.0: `Derivative(f(0), 0)`, a derivative in the number 0.
    // SymPy 1.14: `f(x).diff(x).subs(x, 0)` is
    // `Subs(Derivative(f(x), x), x, 0)` with `free_symbols` `set()`.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d0 = f_of(&ctx, &[&x]).diff(&x).subs_i64(&x, 0);
    assert_eq!(d0.to_string(), "Subs(Derivative(f(x), x), x, 0)");
    assert!(d0.free_symbols().is_empty());
    assert!(d0.has_unevaluated());
    assert_eq!(d0.expr_type(), ExprType::Unevaluated);
    // Bound, so it does not depend on x (SymPy: `diff(_, x)` is 0).
    assert_eq!(d0.diff(&x), ctx.int(0));
    // SymPy: `latex(_)` is `\left. \frac{d}{d x} f{\left(x \right)}
    // \right|_{\substack{ x=0 }}`.
    let tex = d0.to_latex();
    assert!(
        tex.starts_with(r"\left. ") && tex.ends_with(r"\right|_{x=0}"),
        "{tex}"
    );
    // Numeric evaluation is refused, as for any unevaluated node.
    assert!(d0.eval_f64().is_err());
}

#[test]
fn a_derivative_at_a_point_differentiates_what_it_can() {
    // SymPy 1.14: `(x**2*f(x)).diff(x).subs(x, 0)` is `0`, and
    // `Subs(Derivative(sin(x), x), x, 0).doit()` is `1`.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let fx = f_of(&ctx, &[&x]);
    assert_eq!((x.powi(2) * &fx).diff(&x).subs_i64(&x, 0), ctx.int(0));
    let formal = x.sin().formal_diff(&x);
    assert_eq!(formal.subs_i64(&x, 0).eval(), ctx.int(1));
    // A defined function substituted by `replace` (structural), then
    // `eval_derivatives` (SymPy's `doit`).
    let sin_x = x.sin();
    let at0 = fx.diff(&x).subs_i64(&x, 0);
    let defined = at0.replace(|e| if e == fx { Some(sin_x.clone()) } else { None });
    assert_eq!(defined.eval_derivatives().eval(), ctx.int(1));
}

#[test]
fn renaming_the_variable_of_a_derivative_or_integral_renames_it() {
    // SymPy 1.14: `f(x).diff(x).subs(x, y)` is `Derivative(f(y), y)`;
    // `Integral(x**2*y, x).subs(x, t)` is `Integral(t**2*y, t)`.
    let ctx = Context::new();
    let (x, y, t) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("t"));
    let d = f_of(&ctx, &[&x]).diff(&x).subs(&x, &y);
    assert_eq!(d, f_of(&ctx, &[&y]).diff(&y));
    let i = ctx.parse("Integral(x^2*y, x)").unwrap();
    assert_eq!(i.subs(&x, &t), ctx.parse("Integral(t^2*y, t)").unwrap());
    // To a symbol the operand already has: evaluation, not a rename.
    // (SymPy renames here, `Integral(y*f(y), y)`, which changes the value.)
    let iy = i.subs(&x, &y);
    assert_eq!(iy.to_string(), "Subs(Integral(y*x^2, x), x, y)");
}

#[test]
fn an_indefinite_integral_at_a_point_is_a_subs_node() {
    // Up to 0.28.0: `Integral(1/9*y, 1/3)`.  SymPy 1.14:
    // `Integral(x**2*y, x).subs(x, Rational(1, 3))` is
    // `Integral(x**2*y, (x, 1/3))` (the antiderivative at 1/3).
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let i = ctx.parse("Integral(x^2*y, x)").unwrap();
    let at = i.subs(&x, &ctx.rational(1, 3));
    assert_eq!(at.to_string(), "Subs(Integral(y*x^2, x), x, 1/3)");
    assert_eq!(at.free_symbols(), vec![y.clone()]);
    // The point is substituted like any outer operand.
    let a = ctx.symbol("a");
    let at_a = i.subs(&x, &(&a + ctx.rational(1, 3)));
    assert_eq!(at_a.subs_i64(&a, 0), at);
    // Display round-trips through the parser.
    assert_eq!(ctx.parse(&at.to_string()).unwrap(), at);
}

#[test]
fn a_substitution_that_would_bring_in_the_derivative_variable_is_a_subs_node() {
    // Up to 0.28.0: `Derivative(f(x, x), x)`, the total derivative.
    // SymPy 1.14: `Derivative(f(x, y), x).subs(y, x)` is
    // `Subs(Derivative(f(x, y), x), y, x)`; with f(a, b) = a*b**2, `.doit()`
    // is `x**2` (∂₁ only), where the total derivative of x³ is 3x².
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let d = f_of(&ctx, &[&x, &y]).diff(&x).subs(&y, &x);
    assert_eq!(d.to_string(), "Subs(Derivative(f(x_1, x), x_1), x_1, x)");
    let x1 = ctx.symbol("x_1");
    let node = f_of(&ctx, &[&x1, &x]);
    let concrete = &x1 * x.powi(2);
    let defined = d.replace(|e| {
        if e == node {
            Some(concrete.clone())
        } else {
            None
        }
    });
    assert_eq!(defined.eval_derivatives(), x.powi(2));
}

#[test]
fn the_derivative_of_a_subs_node_follows_the_chain_rule() {
    // SymPy 1.14: `diff(f(x).diff(x).subs(x, y**2), y)` is
    // `2*y*Subs(Derivative(f(x), (x, 2)), x, y**2)`.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let s = f_of(&ctx, &[&x]).diff(&x).subs(&x, &y.powi(2));
    assert_eq!(s.to_string(), "Subs(Derivative(f(x), x), x, y^2)");
    assert_eq!(
        s.diff(&y).to_string(),
        "2*y*Subs(Derivative(Derivative(f(x), x), x), x, y^2)"
    );
}

#[test]
fn a_partial_derivative_in_a_non_symbol_argument_is_a_subs_node() {
    // Up to 0.28.0: `2*x*Derivative(f(x^2), x^2)`, and at x = 1
    // `2*Derivative(f(1), 1)`.  SymPy 1.14: `f(x**2).diff(x)` is
    // `2*x*Subs(Derivative(f(_xi_1), _xi_1), _xi_1, x**2)` and `.subs(x, 1)`
    // is `2*Subs(Derivative(f(_xi_1), _xi_1), _xi_1, 1)`.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d = f_of(&ctx, &[&x.powi(2)]).diff(&x);
    assert_eq!(d.to_string(), "2*x*Subs(Derivative(f(_xi), _xi), _xi, x^2)");
    assert_eq!(
        d.subs_i64(&x, 1).to_string(),
        "2*Subs(Derivative(f(_xi), _xi), _xi, 1)"
    );
}

#[test]
fn the_derivative_of_f_x_x_is_not_counted_twice() {
    // Up to 0.28.0: `2*Derivative(f(x, x), x)`, which for f(a, b) = a*b²
    // is 6x².  SymPy 1.14: `f(x, x).diff(x)` is
    // `Subs(Derivative(f(_xi_1, x), _xi_1), _xi_1, x) + Subs(Derivative(f(x,
    // _xi_2), _xi_2), _xi_2, x)`, and with f = Lambda((a, t), a*t**2)
    // `.doit()` is `3*x**2` = `diff(x**3, x)`.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d = f_of(&ctx, &[&x, &x]).diff(&x);
    assert_eq!(
        d.to_string(),
        "Subs(Derivative(f(_xi, x), _xi), _xi, x) + Subs(Derivative(f(x, _xi), _xi), _xi, x)"
    );
    let xi = ctx.symbol("_xi");
    let (slot0, slot1) = (f_of(&ctx, &[&xi, &x]), f_of(&ctx, &[&x, &xi]));
    let (v0, v1) = (&xi * x.powi(2), &x * xi.powi(2));
    let defined = d.replace(|e| {
        if e == slot0 {
            Some(v0.clone())
        } else if e == slot1 {
            Some(v1.clone())
        } else {
            None
        }
    });
    assert_eq!(defined.eval_derivatives(), x.powi(2) * 3);
}

#[test]
fn the_laplace_transform_of_a_second_derivative_evaluates_f_prime_at_zero() {
    // L{g″} = s²G − s·g(0) − g′(0) needs g′(0).  Up to 0.28.0,
    // `Derivative(sin(t), t).subs(t, 0)` was `Derivative(sin(0), 0)` and the
    // transform of the formal (sin t)″ was `s^2/(s^2 + 1) -
    // Derivative(sin(0), 0)`.  SymPy 1.14: `laplace_transform(-sin(t), t, s,
    // noconds=True)` is `-1/(s**2 + 1)`.
    let ctx = Context::new();
    let (t, s) = (ctx.symbol("t"), ctx.symbol("s"));
    let l = t.sin().formal_diff(&t).formal_diff(&t).laplace(&t, &s);
    assert!(!l.has_unevaluated(), "{l}");
    let expected = -(ctx.int(1) / (s.powi(2) + 1));
    assert_eq!((&l - &expected).simplify(), ctx.int(0), "{l}");
}

// ── RootOf records its variable ─────────────────────────────────────────

#[test]
fn a_parametric_root_of_binds_its_variable_only() {
    // Up to 0.28.0: `RootOf(x^5 - a*x + 1, 0).subs(x, 1/3)` was
    // `RootOf(-1/3*a + 244/243, 0)`.  SymPy 1.14 refuses
    // `CRootOf(x**5 - a*x + 1, 0)` ("only univariate polynomials are
    // allowed"); `CRootOf(x**5 - 2*x + 1, 0).evalf(20)` is
    // `-1.2906488013467096224`.
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let r = ctx.parse("RootOf(x^5 - a*x + 1, x, 0)").unwrap();
    assert_eq!(r.to_string(), "RootOf(x^5 - a*x + 1, x, 0)");
    assert_eq!(r.free_symbols(), vec![a.clone()]);
    assert_eq!(r.subs(&x, &ctx.rational(1, 3)), r);
    let at2 = r.subs_i64(&a, 2);
    assert_eq!(at2.to_string(), "RootOf(x^5 - 2*x + 1, 0)");
    let v = at2.eval_f64().unwrap();
    assert!((v + 1.290_648_801_346_709_6).abs() < 1e-14, "{v}");
    // The two-argument form names no variable for such a polynomial.
    assert!(ctx.parse("RootOf(x^5 - a*x + 1, 0)").is_err());
    // The variable can be any symbol of the polynomial.
    let ra = ctx.parse("RootOf(x^5 - a*x + 1, a, 0)").unwrap();
    assert_eq!(ra.free_symbols(), vec![x.clone()]);
}

#[test]
fn root_of_round_trips_through_display_and_tree() {
    // Display, parse and `ExprTree` write the variable exactly when the
    // polynomial has other symbols, so a univariate `RootOf` prints and
    // serialises as before.
    let ctx = Context::new();
    for src in ["RootOf(x^5 - x + 1, 0)", "RootOf(x^5 - a*x + 1, x, 0)"] {
        let r = ctx.parse(src).unwrap();
        assert_eq!(r.to_string(), src);
        assert_eq!(ctx.from_tree(&r.to_tree()), r, "{src}");
        let json = r.to_json().unwrap();
        assert_eq!(json.contains("\"var\""), src.contains(", x, "), "{json}");
    }
    // A tree written before the variable existed reads the only symbol.
    let old = r#"{"type":"RootOf","poly":{"type":"Add","terms":[{"type":"Num","numer":"1","denom":"1"},{"type":"Pow","base":{"type":"Symbol","name":"x"},"exp":{"type":"Num","numer":"5","denom":"1"}},{"type":"Mul","factors":[{"type":"Num","numer":"-1","denom":"1"},{"type":"Symbol","name":"x"}]}]},"index":{"type":"Num","numer":"0","denom":"1"}}"#;
    let tree: symplex::tree::ExprTree = serde_json::from_str(old).unwrap();
    assert_eq!(
        ctx.from_tree(&tree),
        ctx.parse("RootOf(x^5 - x + 1, 0)").unwrap()
    );
}

// ── subs_algebraic and try_replace respect binders ──────────────────────

#[test]
fn subs_algebraic_leaves_a_bound_variable_alone() {
    // Up to 0.28.0: `Sum(k*y, x=0..3)`.  SymPy 1.14:
    // `Sum(k*x**2, (x, 0, 3)).subs(x**2, y)` is unchanged; `.doit()` is
    // `14*k`.
    let ctx = Context::new();
    let (x, y, k) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("k"));
    let s = ctx.parse("Sum(k*x^2, x, 0, 3)").unwrap();
    assert_eq!(s.subs_algebraic(&x.powi(2), &y), s);
    // Free occurrences are still rewritten, inside and outside.
    let e = ctx.parse("x^4 + Sum(k*x^2, j, 0, 3)").unwrap();
    assert_eq!(
        e.subs_algebraic(&x.powi(2), &y),
        ctx.parse("y^2 + Sum(k*y, j, 0, 3)").unwrap()
    );
    let _ = k;
}

#[test]
fn try_replace_refuses_a_non_symbol_bound_variable() {
    // Up to 0.28.0: `Sum(1/9*k, 1/3=0..3)`.  SymPy 1.14:
    // `Sum(k*x**2, (x, 0, 3)).xreplace({x: Rational(1, 3)})` raises
    // `ValueError: Invalid limits given`, and `.xreplace({x: y})` is
    // `Sum(k*y**2, (y, 0, 3))`.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let third = ctx.rational(1, 3);
    let s = ctx.parse("Sum(k*x^2, x, 0, 3)").unwrap();
    let err = s.try_replace(|e| if e == x { Some(third.clone()) } else { None });
    assert!(
        matches!(err, Err(SymplexError::InvalidArgument { .. })),
        "{err:?}"
    );
    let renamed = s.try_replace(|e| if e == x { Some(y.clone()) } else { None });
    assert_eq!(renamed.unwrap(), ctx.parse("Sum(k*y^2, y, 0, 3)").unwrap());
    // The same for a derivative's variable.
    let fx = f_of(&ctx, &[&x]).diff(&x);
    assert!(
        fx.try_replace(|e| if e == x { Some(third.clone()) } else { None })
            .is_err()
    );
}

// ── dependence tests respect binders ────────────────────────────────────

#[test]
fn laplace_and_z_transforms_treat_a_root_of_as_a_constant() {
    // Up to 0.28.0: `LaplaceTransform(RootOf(t^5 - t + 1, 0), t, s)` and
    // "cannot transform RootOf(n^5 - n + 1, 0)".  SymPy 1.14:
    // `laplace_transform(CRootOf(t**5 - t + 1, 0), t, s, noconds=True)` is
    // `CRootOf(t**5 - t + 1, 0)/s`; the Z transform of a constant c is
    // `summation(c*z**(-n), (n, 0, oo))` = c·z/(z − 1).
    let ctx = Context::new();
    let (t, s, n, z) = (
        ctx.symbol("t"),
        ctx.symbol("s"),
        ctx.symbol("n"),
        ctx.symbol("z"),
    );
    let rt = ctx.parse("RootOf(t^5 - t + 1, 0)").unwrap();
    assert_eq!(rt.laplace(&t, &s), &rt / &s);
    let rn = ctx.parse("RootOf(n^5 - n + 1, 0)").unwrap();
    let zt = rn.z_transform(&n, &z).unwrap();
    let expected = &rn * &z / (&z - 1);
    assert_eq!((&zt - &expected).simplify(), ctx.int(0), "{zt}");
}

#[test]
fn summation_treats_a_root_of_in_the_index_as_a_constant() {
    // Up to 0.28.0: `k*(-RootOf(x^5 - x + 1, 0) + (n + 1)*RootOf(x^5 - x +
    // 1, 0))`.  SymPy 1.14: `summation(CRootOf(x**5 - x + 1, 0)*k, (x, 1,
    // n))` is `k*n*CRootOf(t**5 - t + 1, 0)`.
    let ctx = Context::new();
    let (x, k, n) = (ctx.symbol("x"), ctx.symbol("k"), ctx.symbol("n"));
    let r = ctx.parse("RootOf(x^5 - x + 1, 0)").unwrap();
    assert_eq!((&r * &k).summation(&x, &ctx.int(1), &n), &k * &n * &r);
}

// ── Part B: Matrix / ExactMatrix constructors return Result ─────────────

#[test]
fn matrix_constructors_refuse_an_empty_shape() {
    // Up to 0.28.0 each of these panicked ("dimensions must be
    // positive"); SymPy 1.14 has empty matrices (`zeros(0, 3).shape` is
    // `(0, 3)`), symplex has none, and reports the shape as an error.
    let ctx = Context::new();
    let one = ctx.int(1);
    let invalid =
        |r: Result<Matrix, SymplexError>| matches!(r, Err(SymplexError::InvalidArgument { .. }));
    assert!(invalid(Matrix::zeros(&ctx, 0, 3)));
    assert!(invalid(Matrix::zeros(&ctx, 3, 0)));
    assert!(invalid(Matrix::identity(&ctx, 0)));
    assert!(invalid(Matrix::from_fn(0, 2, |_, _| one.clone())));
    assert!(invalid(Matrix::row_vector(vec![])));
    assert!(invalid(Matrix::col_vector(vec![])));
    assert!(invalid(Matrix::diag(&[])));
    // Valid shapes are unchanged.
    assert_eq!(
        Matrix::identity(&ctx, 2).unwrap(),
        matrix![ctx, [1, 0], [0, 1]]
    );
    assert_eq!(
        Matrix::diag(&[one.clone(), ctx.int(2)]).unwrap(),
        matrix![ctx, [1, 0], [0, 2]]
    );
    assert_eq!(
        Matrix::row_vector(vec![one.clone()]).unwrap().shape(),
        (1, 1)
    );
}

#[test]
fn matrix_index_accessors_have_checked_twins() {
    // `get`/`get_mut`/`row`/`col` still panic out of range, like `Vec`
    // indexing (they are the bodies of `m[(i, j)]`); the `try_` twins do
    // not.
    let ctx = Context::new();
    let mut m = matrix![ctx, [1, 2], [3, 4]];
    assert_eq!(m.try_get(1, 1), Some(&ctx.int(4)));
    assert!(m.try_get(2, 0).is_none());
    assert_eq!(m.try_row(1), Some(&[ctx.int(3), ctx.int(4)][..]));
    assert!(m.try_row(2).is_none());
    assert_eq!(m.try_col(0), Some(vec![ctx.int(1), ctx.int(3)]));
    assert!(m.try_col(2).is_none());
    *m.try_get_mut(0, 0).unwrap() = ctx.int(9);
    assert_eq!(m, matrix![ctx, [9, 2], [3, 4]]);
    assert!(m.try_get_mut(0, 2).is_none());
    // The exact cache is invalidated by `try_get_mut`, as by `get_mut`.
    assert_eq!(m.det().unwrap(), ctx.int(30));
}

#[test]
fn exact_matrix_constructors_refuse_an_empty_shape() {
    // Up to 0.28.0 each of these panicked.
    use symplex::linprog::q;
    use symplex::matrix::{QMatrix, ZMatrix};
    assert!(QMatrix::zeros(0, 1).is_err());
    assert!(QMatrix::identity(0).is_err());
    assert!(QMatrix::from_fn(1, 0, |_, _| q(0, 1)).is_err());
    assert!(QMatrix::diag(&[]).is_err());
    assert!(QMatrix::row_vector(vec![]).is_err());
    assert!(ZMatrix::col_vector(vec![]).is_err());
    let mut m = QMatrix::identity(2).unwrap();
    assert_eq!(m.try_col(1), Some(vec![q(0, 1), q(1, 1)]));
    assert!(m.try_row(2).is_none());
    *m.try_get_mut(1, 0).unwrap() = q(5, 1);
    assert_eq!(m[(1, 0)], q(5, 1));
    assert!(m.try_get_mut(2, 2).is_none());
}

#[test]
fn a_formal_derivative_of_a_dependent_symbol_stays_formal_at_a_point() {
    // `y.formal_diff(x)` stands for y′(x) (the ODE convention), so at x = 0
    // it is y′(0), `Subs(Derivative(y, x), x, 0)` — not `diff(y, x)` = 0.
    // Up to 0.28.0 it was `Derivative(y, 0)`.  A formal derivative whose
    // operand depends on x alone is differentiated where `diff` makes
    // progress: SymPy 1.14 `(x**2*f(x)).diff(x).subs(x, 0)` is `0`.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let dy0 = y.formal_diff(&x).subs_i64(&x, 0);
    assert_eq!(dy0.to_string(), "Subs(Derivative(y, x), x, 0)");
    let fx = f_of(&ctx, &[&x]);
    assert_eq!(
        (x.powi(2) * &fx).formal_diff(&x).subs_i64(&x, 0),
        ctx.int(0)
    );
}
