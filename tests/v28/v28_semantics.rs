//! After 0.28.0 — substitution of a function application under a binder,
//! and checked twins for the remaining panics on caller input.
//!
//! - `subs` refused every replacement whose `old` has a bound variable
//!   free, so `f′(0)`, `Subs(Derivative(f(x), x), x, 0)`, with `f(x) ↦
//!   sin(x)` came back unchanged, as did `∫₀¹ f(x) dx` with `f(x) ↦ x²` and
//!   `Σ_k f(k)` with `f(k) ↦ k²`.  An application of an undefined function
//!   is now read as the definition `f := λx. new`, which means the same in
//!   every scope; anything else in the bound variable stays shadowed.
//! - Renaming a binder or a derivative variable to avoid a capture kept
//!   the `new` of a replacement whose `old` mentions the variable
//!   unrenamed: `d/dx (y·f(x))` at `x = 0` with `f(x) ↦ sin(x), y ↦ x` was
//!   `0`.
//! - `MultiPoly::degree_in` / `partial_derivative` / `eval_var` panicked on
//!   a variable index out of range, and `var` / `substitute` /
//!   `s_polynomial` had no checked form.
//!
//! Reference values cite the SymPy 1.14 call they come from.

use num_bigint::BigInt;
use num_rational::Ratio;
use symplex::multipoly::{GrevLex, MultiPoly, s_polynomial, try_s_polynomial};
use symplex::prelude::*;
use symplex::tree::ExprTree;

fn f_of(ctx: &Context, args: &[&Ex]) -> Ex {
    ctx.apply("f", args).unwrap()
}

fn g_of(ctx: &Context, args: &[&Ex]) -> Ex {
    ctx.apply("g", args).unwrap()
}

/// `f′(0)` as the chain rule leaves it: `Subs(Derivative(f(x), x), x, 0)`.
fn f_prime_at_0(ctx: &Context, x: &Ex) -> Ex {
    f_of(ctx, &[x]).diff(x).subs_i64(x, 0)
}

// ── Part A: function patterns under a binder ────────────────────────────

#[test]
fn a_function_pattern_is_substituted_into_a_subs_body() {
    // Up to 0.28.0: unchanged, still `f`.  SymPy 1.14:
    // `Subs(Derivative(f(x), x), x, 0).subs(f(x), sin(x))` is
    // `Subs(Derivative(sin(x), x), x, 0)` and `.doit()` is `1`.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let fx = f_of(&ctx, &[&x]);
    let s0 = f_prime_at_0(&ctx, &x);
    let s = s0.subs(&fx, &x.sin());
    assert_eq!(s.to_string(), "Subs(Derivative(sin(x), x), x, 0)");
    assert_eq!(s.eval_derivatives().eval(), ctx.int(1));
    // SymPy 1.14: `.subs(f(x), x*y).doit()` is `y`, `.subs(f(x), x).doit()`
    // is `1`, `.subs(f(x), g(x))` is `Subs(Derivative(g(x), x), x, 0)`.
    let s = s0.subs(&fx, &(&x * &y));
    assert_eq!(s.to_string(), "Subs(Derivative(x*y, x), x, 0)");
    assert_eq!(s.eval_derivatives(), y);
    assert_eq!(s0.subs(&fx, &x).eval_derivatives(), ctx.int(1));
    let gx = g_of(&ctx, &[&x]);
    assert_eq!(
        s0.subs(&fx, &gx).to_string(),
        "Subs(Derivative(g(x), x), x, 0)"
    );
    // The same order-independence as without the binder: substituting
    // before evaluating at 0 agrees.
    assert_eq!(
        fx.diff(&x).subs(&fx, &x.sin()).subs_i64(&x, 0).eval(),
        ctx.int(1)
    );
}

#[test]
fn the_point_and_the_bound_variable_of_a_subs_are_handled_as_before() {
    let ctx = Context::new();
    let (x, y, p) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("p"));
    let fx = f_of(&ctx, &[&x]);
    // The point is an outer operand.  SymPy 1.14:
    // `Subs(Derivative(f(x), x), x, 0).subs(0, 1)` is
    // `Subs(Derivative(f(x), x), x, 1)`.
    let s0 = f_prime_at_0(&ctx, &x);
    assert_eq!(
        s0.subs(&ctx.int(0), &ctx.int(1)).to_string(),
        "Subs(Derivative(f(x), x), x, 1)"
    );
    // The bound variable is not free.  SymPy 1.14:
    // `Subs(Derivative(f(x), x), x, p + 1).subs(x, 1)` is unchanged and
    // `Subs(Derivative(f(x), x), x, x + 1).subs(x, y)` is
    // `Subs(Derivative(f(y), y), y, y + 1)` (the same node up to the
    // dummy's name).
    let sp = fx.diff(&x).subs(&x, &(&p + 1));
    assert_eq!(sp.to_string(), "Subs(Derivative(f(x), x), x, p + 1)");
    assert_eq!(sp.subs_i64(&x, 1), sp);
    let sx = fx.diff(&x).subs(&x, &(&x + 1));
    assert_eq!(
        sx.subs(&x, &y).to_string(),
        "Subs(Derivative(f(x), x), x, y + 1)"
    );
    // A free symbol of the point and of the body.  SymPy 1.14:
    // `Subs(Derivative(f(x, y), x), x, y).subs(y, 2)` is
    // `Subs(Derivative(f(x, 2), x), x, 2)`.
    let s5 = f_of(&ctx, &[&x, &y]).diff(&x).subs(&x, &y);
    assert_eq!(s5.to_string(), "Subs(Derivative(f(x, y), x), x, y)");
    assert_eq!(
        s5.subs_i64(&y, 2).to_string(),
        "Subs(Derivative(f(x, 2), x), x, 2)"
    );
    // ... and one that would be captured: the dummy is renamed.  (SymPy
    // 1.14 gives `Subs(Derivative(f(x, y), x), (y, x), (x, x))`.)
    assert_eq!(
        s5.subs(&y, &x).to_string(),
        "Subs(Derivative(f(x_1, x), x_1), x_1, x)"
    );
}

#[test]
fn a_function_pattern_is_substituted_into_sum_integral_and_limit_bodies() {
    let ctx = Context::new();
    let (x, y, k, n) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("k"),
        ctx.symbol("n"),
    );
    let (zero, one) = (ctx.int(0), ctx.int(1));
    let fx = f_of(&ctx, &[&x]);
    // Up to 0.28.0: unchanged.  SymPy 1.14:
    // `Integral(f(x), (x, 0, 1)).subs(f(x), x**2)` is
    // `Integral(x**2, (x, 0, 1))`, `.doit()` `1/3`.
    let i = fx.definite_integral_node(&x, &zero, &one);
    let s = i.subs(&fx, &x.powi(2));
    assert_eq!(s.to_string(), "Integral(x^2, x, 0, 1)");
    assert_eq!(s.eval_integrals(), ctx.rational(1, 3));
    // The free `f(x)` outside and the bound one inside agree.  SymPy 1.14:
    // `(x*Integral(f(x), (x, 0, 1))).subs(f(x), x**2)` is
    // `x*Integral(x**2, (x, 0, 1))`.
    let s = (&fx * &i).subs(&fx, &x.powi(2));
    assert_eq!(s.to_string(), "x^2*Integral(x^2, x, 0, 1)");
    // SymPy 1.14: `Sum(f(k), (k, 0, n)).subs(f(k), k**2)` is
    // `Sum(k**2, (k, 0, n))`, and with `x*k`, `Sum(k*x, (k, 0, n))`.
    let fk = f_of(&ctx, &[&k]);
    let sum = Ex::symbolic_sum(&fk, &k, &zero, &n);
    assert_eq!(sum.subs(&fk, &k.powi(2)).to_string(), "Sum(k^2, k=0..n)");
    assert_eq!(sum.subs(&fk, &(&x * &k)).to_string(), "Sum(k*x, k=0..n)");
    // SymPy 1.14: `Limit(f(x)*y, x, 0).subs(f(x), sin(x)/x)` is
    // `Limit(y*sin(x)/x, x, 0, dir='+')`, `.doit()` `y`.
    let lim = ctx.from_tree(&ExprTree::Limit {
        body: Box::new((&fx * &y).to_tree()),
        var: Box::new(x.to_tree()),
        point: Box::new(zero.to_tree()),
    });
    let s = lim.subs(&fx, &(x.sin() / &x));
    assert_eq!(
        s,
        ctx.from_tree(&ExprTree::Limit {
            body: Box::new((&y * x.sin() / &x).to_tree()),
            var: Box::new(x.to_tree()),
            point: Box::new(zero.to_tree()),
        })
    );
    // A variable slot, as before.  SymPy 1.14:
    // `Derivative(f(x), x).subs(f(x), sin(x))` is `Derivative(sin(x), x)`.
    assert_eq!(fx.diff(&x).subs(&fx, &x.sin()).eval_derivatives(), x.cos());
    // `ConditionSet` binds the same way.  (SymPy 1.14 keeps
    // `ConditionSet(x, Eq(f(x), 0), Reals).subs(f(x), x**2 - 1)` unchanged:
    // its `ConditionSet._eval_subs` substitutes into the condition only
    // when `new` is a symbol or an applied undefined function.)
    let cs = ctx.from_tree(&ExprTree::ConditionSet {
        var: Box::new(x.to_tree()),
        condition: Box::new(fx.to_tree()),
    });
    let s = cs.subs(&fx, &(x.powi(2) - 1));
    assert_eq!(s.free_symbols(), Vec::<Ex>::new(), "{s}");
    assert!(!s.to_string().contains('f'), "{s}");
}

#[test]
fn a_defined_function_or_a_non_argument_pattern_stays_shadowed() {
    let ctx = Context::new();
    let (x, k, n) = (ctx.symbol("x"), ctx.symbol("k"), ctx.symbol("n"));
    let (zero, one) = (ctx.int(0), ctx.int(1));
    // SymPy 1.14: `Integral(sin(x), (x, 0, 1)).subs(sin(x), cos(x))` is
    // unchanged: `sin` is not being defined, and the `sin(x)` of the
    // pattern is the outer `x`.
    let i = x.sin().definite_integral_node(&x, &zero, &one);
    assert_eq!(i.subs(&x.sin(), &x.cos()), i);
    // A library function carried as an application is defined too.
    let j0 = ctx.parse("besselj(0, x)").unwrap();
    let i = j0.definite_integral_node(&x, &zero, &one);
    assert_eq!(i.subs(&j0, &x), i);
    // `new` in the bound variable but not through an argument slot:
    // SymPy 1.14 raises `ValueError: substitution cannot create dummy
    // dependencies` for `Integral(f(x + 1), (x, 0, 1)).subs(f(x + 1),
    // sin(x))`; symplex leaves the body alone.  With a `new` free of `x`
    // it is a definition (`f := λu. 5`): SymPy 1.14 gives
    // `Integral(5, (x, 0, 1))`.
    let fx1 = f_of(&ctx, &[&(&x + 1)]);
    let i = fx1.definite_integral_node(&x, &zero, &one);
    assert_eq!(i.subs(&fx1, &x.sin()), i);
    assert_eq!(i.subs(&fx1, &ctx.int(5)), ctx.int(5));
    // An `old` free of the index is an ordinary replacement, renamed
    // rather than captured.  (SymPy 1.14 raises the same `ValueError` for
    // `Sum(f(x), (k, 0, n)).subs(f(x), k)`.)
    let fx = f_of(&ctx, &[&x]);
    let s = Ex::symbolic_sum(&fx, &k, &zero, &n).subs(&fx, &k);
    assert_eq!(s.to_string(), "Sum(k, k_1=0..n)");
}

#[test]
fn a_function_pattern_is_renamed_with_the_binder_it_is_under() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let (zero, one) = (ctx.int(0), ctx.int(1));
    let fx = f_of(&ctx, &[&x]);
    // ∫₀¹ f(x)·y dx with f(x) ↦ sin(x), y ↦ x (the outer x) at once:
    // x·(1 − cos 1).  Up to 0.28.0 `Integral(x*f(x_1), x_1, 0, 1)` (the
    // pattern shadowed).  SymPy 1.14's `Integral(f(x)*y, (x, 0,
    // 1)).subs({f(x): sin(x), y: x}, simultaneous=True)` captures:
    // `Integral(x*sin(x), (x, 0, 1))`, `sin(1) - cos(1)`.
    let i = (&fx * &y).definite_integral_node(&x, &zero, &one);
    let s = i.subs_map(&[(&fx, &x.sin()), (&y, &x)]);
    assert_eq!(s.to_string(), "Integral(x*sin(x_1), x_1, 0, 1)");
    let v = s.eval_integrals().subs_i64(&x, 2).eval_f64().unwrap();
    let want = 2.0 * (1.0 - 1f64.cos());
    assert!((v - want).abs() < 1e-14, "{v} vs {want}");
    // A nested binder of the same variable keeps its own `x`: the inner
    // `f(x)` is still the pattern, not the renamed `x_1`.
    let inner = fx.definite_integral_node(&x, &zero, &one);
    let nested = (&fx * &y + inner).definite_integral_node(&x, &zero, &one);
    let s = nested.subs_map(&[(&fx, &x.sin()), (&y, &x)]);
    assert_eq!(
        s.to_string(),
        "Integral(x*sin(x_1) + Integral(sin(x), x, 0, 1), x_1, 0, 1)"
    );
    let v = s.eval_integrals().subs_i64(&x, 2).eval_f64().unwrap();
    let want = 3.0 * (1.0 - 1f64.cos());
    assert!((v - want).abs() < 1e-14, "{v} vs {want}");
}

#[test]
fn a_derivative_evaluated_at_a_point_renames_the_pattern_with_its_variable() {
    let ctx = Context::new();
    let (x, y, t) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("t"));
    let fx = f_of(&ctx, &[&x]);
    // d/dx (y·f(x)) at x = 0 with f(x) ↦ sin(x), y ↦ x at once: y·cos(0)
    // with y ↦ x, i.e. x.  Up to 0.28.0
    // `Subs(Derivative(x*sin(x), x_1), x_1, 0)`, which is 0.  (SymPy 1.14's
    // `Derivative(y*f(x), x).subs({x: 0, f(x): sin(x), y: x},
    // simultaneous=True)` captures the same way: `Subs(Derivative(x*sin(x),
    // x), x, 0)`, `.doit()` `0`.)
    let d = (&fx * &y).formal_diff(&x);
    let s = d.subs_map(&[(&x, &ctx.int(0)), (&fx, &x.sin()), (&y, &x)]);
    assert_eq!(s.to_string(), "Subs(Derivative(x*sin(x_1), x_1), x_1, 0)");
    assert_eq!(s.eval_derivatives().eval(), x);
    // Renaming the variable to a new symbol renames the pattern too, as
    // one substitution after the other does.  Up to 0.28.0
    // `Subs(Derivative(g(x), x), x, t)`.  SymPy 1.14:
    // `Derivative(f(x), x).subs({x: t, f(x): g(x)}, simultaneous=True)` is
    // `Subs(Derivative(g(x), x), x, t)`, `.doit()` `Derivative(g(t), t)`.
    let gx = g_of(&ctx, &[&x]);
    let df = fx.diff(&x);
    let at_once = df.subs_map(&[(&x, &t), (&fx, &gx)]);
    assert_eq!(at_once, g_of(&ctx, &[&t]).diff(&t));
    assert_eq!(at_once, df.subs(&fx, &gx).subs(&x, &t));
    // Still simultaneous: `t ↦ 5` acts on the coefficient `t`, not on the
    // renamed variable.  SymPy 1.14: `Derivative(t*f(x), x).subs({x: t,
    // f(x): g(x), t: 5}, simultaneous=True).doit()` is
    // `5*Derivative(g(t), t)`.
    let five = ctx.int(5);
    let dt = (&fx * &t).formal_diff(&x);
    let s = dt.subs_map(&[(&x, &t), (&fx, &gx), (&t, &five)]);
    assert_eq!(s, g_of(&ctx, &[&t]).diff(&t) * 5);
}

// ── Part B: no panics on caller input ────────────────────────────────

fn q(n: i64) -> Ratio<BigInt> {
    Ratio::from_integer(BigInt::from(n))
}

#[test]
fn multipoly_queries_on_an_absent_variable_are_total() {
    // Up to 0.28.0 each of these panicked ("var_index 2 out of range for 2
    // variables").  A polynomial in x₀, x₁ does not involve x₂: degree 0,
    // derivative 0, unchanged by x₂ ↦ 3 — SymPy 1.14 agrees once the
    // variable is a generator: `Poly(x**2*y, x, y, z).degree(z)` is `0`,
    // `.diff(z)` is `Poly(0, x, y, z)`.
    let [x, y]: [MultiPoly<GrevLex>; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    let p = x.mul(&x).mul(&y).add(&y); // x²y + y
    assert_eq!(p.degree_in(2), 0);
    assert_eq!(p.degree_in(usize::MAX), 0);
    assert_eq!(p.partial_derivative(2), MultiPoly::zero(2));
    assert_eq!(p.eval_var(7, &q(3)), p);
    // In range, unchanged.
    assert_eq!(p.degree_in(0), 2);
    assert_eq!(p.partial_derivative(0), x.mul(&y).scale(&q(2)));
    assert_eq!(p.eval_var(1, &q(3)), x.mul(&x).scale(&q(3)) + 3);
}

#[test]
fn multipoly_index_methods_have_checked_twins() {
    type P = MultiPoly<GrevLex>;
    assert_eq!(P::try_var(2, 1), Some(P::var(2, 1)));
    assert_eq!(P::try_var(2, 2), None);
    assert_eq!(P::try_var(0, 0), None);
    let [x, y]: [P; 2] = [P::var(2, 0), P::var(2, 1)];
    let p = x.mul(&y).add(&x.mul(&x)); // xy + x²
    assert_eq!(p.try_substitute(0, &q(3)), Some(p.substitute(0, &q(3))));
    assert_eq!(p.try_substitute(2, &q(3)), None);
    assert_eq!(P::zero(0).try_substitute(0, &q(1)), None);
    // S(x² + y, xy) = y²: the twin agrees with the function where that
    // succeeds, and refuses mismatched rings and exponent overflow.
    let f = x.mul(&x).add(&y);
    let g = x.mul(&y);
    assert_eq!(try_s_polynomial(&f, &g), Some(s_polynomial(&f, &g)));
    assert_eq!(try_s_polynomial(&f, &P::zero(2)), Some(P::zero(2)));
    assert_eq!(try_s_polynomial(&f, &P::var(3, 0)), None);
    // f = y^MAX, g = x² + y: L = x²·y^MAX, and (L/x²)·y = y^(MAX+1).
    let f = P::monomial(q(1), vec![0, u32::MAX]);
    let g = x.mul(&x).add(&y);
    assert_eq!(try_s_polynomial(&f, &g), None);
    assert!(std::panic::catch_unwind(|| s_polynomial(&f, &g)).is_err());
}

#[test]
fn an_empty_symbol_name_is_refused_by_the_checked_constructor() {
    // `Context::symbol` and its alias `var` panic on "" (documented);
    // `try_symbol` is the checked form.
    let ctx = Context::new();
    assert!(ctx.try_symbol("").is_err());
    assert_eq!(ctx.try_symbol("x").unwrap(), ctx.var("x"));
}
