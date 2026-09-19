//! symplex 0.2 base-layer fixes — bound variables are not free symbols.
//!
//! `RootOf(λ³ − λ − 1, 0)` is a constant: the polynomial variable is
//! bound.  Likewise the index of a formal `Sum`/`Product` is bound in the
//! body (but not in the bounds).  `is_constant` / `free_symbols` must
//! agree with that.

use symplex::prelude::*;

#[test]
fn rootof_from_solver_is_constant() {
    let ctx = Context::new();
    let lam = ctx.symbol("lambda");
    // Irreducible quintic → RootOf solutions.
    let poly = lam.powi(5) - &lam - 1;
    let roots = poly.solve_or_empty(&lam);
    assert_eq!(roots.len(), 5, "{roots:?}");
    for r in &roots {
        assert!(r.to_string().contains("RootOf"), "{r}");
        assert!(r.free_symbols().is_empty(), "{r} has free symbols");
        assert!(r.is_constant(), "{r} should be constant");
    }
    // A RootOf is numerically evaluable — it really is a number.
    let v = roots[0].eval_complex64().unwrap();
    assert!(v.0.is_finite() && v.1.is_finite());
}

#[test]
fn rootof_times_symbol_has_only_that_symbol_free() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let roots = (x.powi(5) - &x - 1).solve_or_empty(&x);
    let e = &roots[1] * &y + 1;
    assert_eq!(e.free_symbols(), vec![y.clone()], "{e}");
    assert!(!e.is_constant());
}

#[test]
fn formal_sum_index_is_bound_bounds_are_free() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    // No closed form → stays a formal Sum node.
    let s = k.sin().summation(&k, &ctx.int(1), &n);
    assert!(s.to_string().contains("Sum"), "{s}");
    assert_eq!(s.free_symbols(), vec![n.clone()], "{s}");
    assert!(!s.is_constant());

    let s10 = k.sin().summation(&k, &ctx.int(1), &ctx.int(10));
    if s10.to_string().contains("Sum") {
        assert!(s10.free_symbols().is_empty(), "{s10}");
        assert!(s10.is_constant(), "{s10}");
    }
    // The constant sum evaluates numerically.
    let v = s10.eval_f64().unwrap();
    let expected: f64 = (1..=10).map(|i| (i as f64).sin()).sum();
    assert!((v - expected).abs() < 1e-9, "{v} vs {expected}");
}

#[test]
fn formal_product_index_is_bound() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let p = (k.sin() + 2).product_over(&k, &ctx.int(1), &n);
    assert!(p.to_string().contains("Product"), "{p}");
    assert_eq!(p.free_symbols(), vec![n.clone()], "{p}");
}

#[test]
fn symbol_free_outside_and_bound_inside_same_expression() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let s = k.sin().summation(&k, &ctx.int(1), &n);
    // k + Σ_{k=1}^{n} sin(k): the leading k is free.
    let e = &k + &s;
    let mut free = e.free_symbols();
    free.sort_by_key(|s| s.to_string());
    assert_eq!(free, vec![k.clone(), n.clone()], "{e}");
}

#[test]
fn integral_and_derivative_variables_stay_free() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(exp(x)) has no elementary or special-function antiderivative in
    // the tables → formal Integral.  (exp(x²) integrates to erfi since 0.9.)
    let i = x.exp().exp().integrate(&x);
    assert!(i.has_unevaluated(), "{i}");
    assert_eq!(i.free_symbols(), vec![x.clone()], "{i}");
    assert!(!i.is_constant());
}
