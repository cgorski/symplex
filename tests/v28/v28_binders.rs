//! After 0.28.0 — substitution respects binders; `apart` classifies roots
//! by value.
//!
//! - `subs` replaced every occurrence of a symbol, bound ones included
//!   (only a `DefiniteIntegral`'s variable was protected):
//!   `RootOf(x⁵ − x + 1, 0)` at `x = 1/3` became `RootOf(163/243, 0)`, which
//!   does not evaluate, and `Σ_{x=0}^{3} k·x` became `Sum(1/3*k, 1/3=0..3)`.
//!   Now only free occurrences are replaced, over one binder table shared
//!   with `free_symbols` (`Sum`, `Product`, `DefiniteIntegral`, `RootSum`,
//!   `ConditionSet`, a univariate `RootOf`, and now `Limit`, `Residue` and
//!   the Laplace transforms).
//! - A replacement that brought a free symbol into the scope of a binder of
//!   the same name was captured: `Σ_{k=0}^{n} x·k` with `x ↦ k` became
//!   `Σ k²`.  The binder is now renamed first (`k_1`).
//! - `d/dx Σ_{k=0}^{x} k` was `Σ 0`, i.e. 0: the limits were ignored.
//! - `apart` took every root without an explicit `i` for a real one, so the
//!   imaginary roots `±√(−1/2 − √5/2)` of `x⁴ + x² − 1` came out as two
//!   complex terms instead of one real quadratic term.
//!
//! Reference values cite the SymPy 1.14 / mpmath 1.3 call they come from.

use symplex::prelude::*;

/// `e` evaluated at `x = p/q` (and `y` at `2`, when given).
fn at(e: &Ex, x: &Ex, p: i64, q: i64) -> num_complex::Complex64 {
    let ctx = e.context();
    e.subs(x, &ctx.rational(p, q)).eval_complex64().unwrap()
}

// ── subs: bound variables are left alone ────────────────────────────────

#[test]
fn a_root_of_a_polynomial_in_x_is_unchanged_by_x_to_a_third() {
    // Up to 0.28.0: `RootOf(163/243, 0)`, then "RootOf polynomial has no
    // variables".  SymPy 1.14: `CRootOf(x**5 - x + 1, 0).subs(x,
    // Rational(1, 3))` is unchanged, and `.evalf(20)` is
    // `-1.1673039782614186843`.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = ctx.parse("RootOf(x^5 - x + 1, 0)").unwrap();
    let s = r.subs(&x, &ctx.rational(1, 3));
    assert_eq!(s, r);
    let v = s.eval_f64().unwrap();
    assert!((v + 1.167_303_978_261_418_7).abs() < 1e-15, "{v}");
}

#[test]
fn the_index_of_a_sum_or_product_is_not_substituted() {
    // Up to 0.28.0: `Sum(1/3*k, 1/3=0..3)` and `Product(1/3*k, 1/3=1..3)`.
    // SymPy 1.14: `Sum(k*x, (x, 0, 3)).subs(x, Rational(1, 3))` and the
    // `Product` are unchanged; `.doit()` gives `6*k` and `6*k**3`.
    let ctx = Context::new();
    let (x, k) = (ctx.symbol("x"), ctx.symbol("k"));
    let third = ctx.rational(1, 3);
    for (src, at_k2) in [("Sum(k*x, x, 0, 3)", 12.0), ("Product(k*x, x, 1, 3)", 48.0)] {
        let e = ctx.parse(src).unwrap();
        let s = e.subs(&x, &third);
        assert_eq!(s, e, "{src}");
        let v = s.subs_i64(&k, 2).eval_f64().unwrap();
        assert_eq!(v, at_k2, "{src} at k = 2");
    }
}

#[test]
fn a_limit_that_mentions_the_index_is_substituted_the_body_is_not() {
    // Up to 0.28.0: `Sum(3*k, 3=0..3)` for the first.  SymPy 1.14:
    // `Sum(k*x, (x, 0, x)).subs(x, 3)` is `Sum(k*x, (x, 0, 3))`, and
    // `(x + Sum(x, (x, 0, 3))).subs(x, Rational(1, 3)).doit()` is `19/3`.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.parse("Sum(k*x, x, 0, x)").unwrap();
    assert_eq!(e.subs_i64(&x, 3), ctx.parse("Sum(k*x, x, 0, 3)").unwrap());

    // The same node `x` free and bound (the arena shares it).
    let e = ctx.parse("x + Sum(x, x, 0, 3)").unwrap();
    let s = e.subs(&x, &ctx.rational(1, 3));
    assert_eq!(s.to_string(), "Sum(x, x=0..3) + 1/3");
    assert!((s.eval_f64().unwrap() - 19.0 / 3.0).abs() < 1e-14);
}

#[test]
fn an_expression_in_the_bound_variable_is_not_matched_under_the_binder() {
    // SymPy 1.14: `Sum(sin(k), (k, 0, n)).subs(sin(k), y)` is unchanged:
    // the `sin(k)` in the body is a function of the bound `k`.
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let y = ctx.symbol("y");
    let e = ctx.parse("Sum(sin(k), k, 0, n)").unwrap();
    assert_eq!(e.subs(&k.sin(), &y), e);
}

#[test]
fn condition_set_and_limit_bind_their_variable() {
    // Up to 0.28.0 `ConditionSet(1/3, sin(1/3) - y)` and
    // `Limit(3*sin(1/3*y), 1/3, 0)`.  SymPy 1.14:
    // `Limit(sin(x*y)/x, x, 0).free_symbols` is `{y}`.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let third = ctx.rational(1, 3);
    let cs = ctx.parse("ConditionSet(x, sin(x) - y)").unwrap();
    assert_eq!(cs.subs(&x, &third), cs);
    let lim = ctx.parse("Limit(sin(x*y)/x, x, 0)").unwrap();
    assert_eq!(lim.subs(&x, &third), lim);
    assert_eq!(lim.free_symbols(), vec![y.clone()]);
    assert_eq!(
        lim.subs_i64(&y, 2),
        ctx.parse("Limit(sin(2*x)/x, x, 0)").unwrap()
    );
}

// ── subs: no capture ────────────────────────────────────────────────────

#[test]
fn substituting_the_name_of_a_bound_variable_renames_the_binder() {
    // Up to 0.28.0: `Sum(k^2, k=0..n)`, which is 14 at n = 3, k = 2.  (SymPy
    // 1.14 captures the same way: `Sum(x*k, (k, 0, n)).subs(x, k)` is
    // `Sum(k**2, (k, 0, n))`.)  The value is
    // `Sum(x*j, (j, 0, 3)).subs(x, 2).doit()` = 12.
    let ctx = Context::new();
    let (x, k, n) = (ctx.symbol("x"), ctx.symbol("k"), ctx.symbol("n"));
    let s = ctx.parse("Sum(x*k, k, 0, n)").unwrap().subs(&x, &k);
    assert_eq!(s.to_string(), "Sum(k*k_1, k_1=0..n)");
    assert_eq!(s.free_symbols().len(), 2, "{s}: k and n are free");
    let v = s.subs_i64(&n, 3).subs_i64(&k, 2).eval_f64().unwrap();
    assert_eq!(v, 12.0);
    // Deterministic: the same name on a second substitution.
    let again = ctx.parse("Sum(x*k, k, 0, n)").unwrap().subs(&x, &k);
    assert_eq!(again, s);

    // Up to 0.28.0: `Integral(k^2, k, 0, 1)` = 1/3 whatever k.
    // SymPy 1.14: `Integral(x*j, (j, 0, 1)).subs(x, 2).doit()` = 1.
    let i = ctx.parse("Integral(x*k, k, 0, 1)").unwrap().subs(&x, &k);
    let v = i.subs_i64(&k, 2).eval_f64().unwrap();
    assert!((v - 1.0).abs() < 1e-14, "{i} at k = 2: {v}");

    // Up to 0.28.0: `ConditionSet(x, -x + sin(x))`, with no free symbol.
    let cs = ctx.parse("ConditionSet(x, sin(x) - y)").unwrap();
    let cs = cs.subs(&ctx.symbol("y"), &x);
    assert_eq!(cs.free_symbols(), vec![x.clone()], "{cs}");
}

#[test]
fn simultaneous_substitution_respects_binders() {
    // `x ↦ k, k ↦ x` at once: the free `k` of the upper limit becomes `x`,
    // the bound `k` stays bound, and the free `x` of the body becomes the
    // outer `k` (which forces the renaming).
    let ctx = Context::new();
    let (x, k) = (ctx.symbol("x"), ctx.symbol("k"));
    let e = ctx.parse("Sum(x*k, k, 0, k)").unwrap();
    let s = e.subs_map(&[(&x, &k), (&k, &x)]);
    assert_eq!(s.to_string(), "Sum(k*k_1, k_1=0..x)");
    // Σ_{j=0}^{3} 2·j = 12 at x = 3, k = 2.
    let v = s.subs_i64(&x, 3).subs_i64(&k, 2).eval_f64().unwrap();
    assert_eq!(v, 12.0);
}

// ── diff over binders ───────────────────────────────────────────────────

#[test]
fn a_sum_whose_limits_depend_on_x_is_not_differentiated_term_by_term() {
    // Up to 0.28.0 `d/dx Σ_{k=0}^{x} k` was `Sum(0, k=0..x)`, i.e. 0, while
    // the sum is x(x + 1)/2.  SymPy 1.14: `Sum(k, (k, 0, x)).diff(x)` is
    // `Derivative(Sum(k, (k, 0, x)), x)`; `Sum(x, (x, 0, 3)).diff(x)` and
    // `Product(x, (x, 1, 3)).diff(x)` are `0`.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d = ctx.parse("Sum(k, k, 0, x)").unwrap().diff(&x);
    assert!(
        d.has_unevaluated() && d.to_string().starts_with("Derivative("),
        "{d}"
    );
    for src in ["Sum(x, x, 0, 3)", "Product(x, x, 1, 3)"] {
        let d = ctx.parse(src).unwrap().diff(&x);
        assert!(d.is_zero_structural(), "d/dx {src} = {d}");
    }
    let d = ctx.parse("Sum(x*k, k, 0, 3)").unwrap().diff(&x);
    assert_eq!(d.eval().to_string(), "6");
}

// ── the integrator ─────────────────────────────────────────────────────────

#[test]
fn a_constant_root_term_of_a_sum_is_integrated() {
    // Up to 0.28.0 a term free of `x` that is not a number, a symbol or a
    // product (a `RootOf`, also one written in `x`) fell through to the
    // unevaluated form: `∫ (x + RootOf(x⁵ − x + 1, 0)) dx` was
    // `x²/2 + Integral(RootOf(x^5 - x + 1, 0), x)`.  In the second, the
    // self-check binds the parameter `a` to a sample value, and the bound
    // `a` of the `RootOf` must stay (0.28.0 renamed it first; `subs` now
    // leaves it alone).  SymPy 1.14:
    // `integrate(a*x + CRootOf(a**5 - a + 1, 0), x)` is
    // `a*x**2/2 + x*CRootOf(x**5 - x + 1, 0)`.
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let r = -1.167_303_978_261_418_7; // CRootOf(x**5 - x + 1, 0).evalf(20)
    for (src, slope_at_third) in [
        ("x + RootOf(x^5 - x + 1, 0)", 1.0 / 3.0 + r),
        ("a*x + RootOf(a^5 - a + 1, 0)", 2.0 / 3.0 + r),
    ] {
        let f = ctx.parse(src).unwrap();
        let big_f = f.integrate(&x);
        assert!(!big_f.has_unevaluated(), "∫ {f} dx = {big_f}");
        let slope = big_f.diff(&x).subs_i64(&a, 2).subs(&x, &ctx.rational(1, 3));
        let slope = slope.eval_f64().unwrap();
        assert!((slope - slope_at_third).abs() < 1e-14, "{big_f}: {slope}");
    }
}

// ── apart: realness by value ────────────────────────────────────────────

#[test]
fn imaginary_roots_without_an_explicit_i_form_a_real_quadratic_term() {
    // x⁴ + x² − 1 has the real roots ±√(√5/2 − 1/2) and the imaginary
    // roots ±√(−1/2 − √5/2).  Up to 0.28.0 each imaginary root had its own
    // complex term `c/(x ∓ √(−1/2·√5 − 1/2))`.  SymPy 1.14:
    // `apart(1/(x**4 + x**2 - 1), x, extension=sqrt(5))` is
    // `2*sqrt(5)/(5*(2*x**2 - sqrt(5) + 1)) - 2*sqrt(5)/(5*(2*x**2 + 1 + sqrt(5)))`.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (src, quadratic) in [
        (
            "1/(x^4 + x^2 - 1)",
            "-sqrt(5)/(5*(x^2 + 1/2*sqrt(5) + 1/2))",
        ),
        // `apart(x**3/(x**4 + 2*x**2 - 1), x, extension=sqrt(2))` has the
        // term `x*(sqrt(2) + 2)/(4*(x**2 + 1 + sqrt(2)))`.
        (
            "x^3/(x^4 + 2*x^2 - 1)",
            "x*(1/4*sqrt(2) + 1/2)/(x^2 + sqrt(2) + 1)",
        ),
    ] {
        let f = ctx.parse(src).unwrap();
        let a = f.partial_fractions(&x);
        let terms = a.args();
        assert_eq!(terms.len(), 3, "{src}: {a}");
        assert!(
            terms.iter().any(|t| t.to_string() == quadratic),
            "{src}: {a} has no term {quadratic}"
        );
        for (p, q) in [(1, 3), (7, 5), (-2, 7)] {
            let (va, vf) = (at(&a, &x, p, q), at(&f, &x, p, q));
            assert!((va - vf).norm() <= 1e-13 * vf.norm(), "{src} at {p}/{q}");
            for t in &terms {
                let v = at(t, &x, p, q);
                assert!(
                    v.im.abs() <= 1e-13 * v.norm(),
                    "{src}: term {t} = {v} at {p}/{q}"
                );
            }
        }
    }
}

#[test]
fn a_cubic_decomposes_into_terms_that_recombine_exactly() {
    // Not a regression (these passed in 0.28.0 too): the realness test by
    // value must keep them exact.  Three real roots written with `i`
    // (casus irreducibilis) now count as real or undecided, never as a
    // conjugate pair; one real root with a complex pair written with `i`
    // keeps its pair term.  Each is checked against the fraction at three
    // points.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["1/(x^3 - 3*x + 1)", "1/(x^3 + x + 1)", "x/(x^3 - x + 1)"] {
        let f = ctx.parse(src).unwrap();
        let a = f.partial_fractions(&x);
        for (p, q) in [(1, 3), (7, 5), (-2, 7)] {
            let (va, vf) = (at(&a, &x, p, q), at(&f, &x, p, q));
            assert!(
                (va - vf).norm() <= 1e-12 * vf.norm(),
                "{src} = {a} at {p}/{q}"
            );
        }
    }
}
