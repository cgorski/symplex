//! symplex 0.2 base-layer fixes — canonical `Mul` construction.
//!
//! * `0 × (±∞ | zoo | nan)` is `nan` regardless of argument order.
//! * Products of numeric radicals never produce a `Mul` nested inside a
//!   `Mul` (which fires `canon_mul`'s canonical-form `debug_assert`).

use symplex::prelude::*;

fn s<T: std::fmt::Display>(e: &T) -> String {
    format!("{e}")
}

fn specials(ctx: &Context) -> Vec<(&'static str, Ex)> {
    vec![
        ("0", ctx.int(0)),
        ("oo", ctx.infinity()),
        ("-oo", ctx.neg_infinity()),
        ("zoo", ctx.complex_infinity()),
        ("nan", ctx.nan()),
    ]
}

#[test]
fn zero_times_infinity_is_nan_in_every_order() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    for (name, v) in specials(&ctx) {
        if name == "0" {
            continue;
        }
        assert_eq!(s(&(&zero * &v)), "nan", "0 * {name}");
        assert_eq!(s(&(&v * &zero)), "nan", "{name} * 0");
    }
}

#[test]
fn products_of_specials_are_commutative() {
    let ctx = Context::new();
    let sp = specials(&ctx);
    for (na, a) in &sp {
        for (nb, b) in &sp {
            let ab = a * b;
            let ba = b * a;
            assert_eq!(ab, ba, "{na} * {nb} = {ab} but {nb} * {na} = {ba}");
        }
    }
}

#[test]
fn products_of_specials_expected_values() {
    let ctx = Context::new();
    let oo = ctx.infinity();
    let moo = ctx.neg_infinity();
    let zoo = ctx.complex_infinity();
    let nan = ctx.nan();
    assert_eq!(s(&(&oo * &oo)), "oo");
    assert_eq!(s(&(&oo * &moo)), "-oo");
    assert_eq!(s(&(&moo * &moo)), "oo");
    assert_eq!(s(&(&oo * &zoo)), "zoo");
    assert_eq!(s(&(&moo * &zoo)), "zoo");
    assert_eq!(s(&(&zoo * &zoo)), "zoo");
    for (name, v) in specials(&ctx) {
        assert_eq!(s(&(&nan * &v)), "nan", "nan * {name}");
        assert_eq!(s(&(&v * &nan)), "nan", "{name} * nan");
    }
}

#[test]
fn zero_times_infinity_with_symbolic_factors_between() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    for (name, v) in specials(&ctx) {
        if name == "0" {
            continue;
        }
        let a = &(&zero * &x) * &v;
        let b = &(&v * &x) * &zero;
        let c = &(&x * &v) * &zero;
        let d = &zero * &(&x * &v);
        assert_eq!(s(&a), "nan", "(0*x)*{name}");
        assert_eq!(s(&b), "nan", "({name}*x)*0");
        assert_eq!(s(&c), "nan", "(x*{name})*0");
        assert_eq!(s(&d), "nan", "0*(x*{name})");
    }
}

#[test]
fn zero_times_zoo_via_inverse_of_zero() {
    // The shape hit by `proptest_algebraic::mul_commutative`: `0 * 0^(-1)`.
    let ctx = Context::new();
    let zero = ctx.int(0);
    let inv0 = zero.powi(-1);
    assert_eq!(s(&inv0), "zoo");
    assert_eq!(s(&(&zero * &inv0)), s(&(&inv0 * &zero)));
    assert_eq!(s(&(&zero * &inv0)), "nan");
}

#[test]
fn zero_times_finite_is_zero_in_every_order() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let finite = vec![
        x.clone(),
        x.sin(),
        &x + 1,
        ctx.pi(),
        ctx.rational(3, 7),
        x.powi(-1),
        (&x + 1).exp(),
    ];
    for f in finite {
        assert_eq!(s(&(&zero * &f)), "0", "0 * {f}");
        assert_eq!(s(&(&f * &zero)), "0", "{f} * 0");
    }
}

#[test]
fn zero_times_unknown_finiteness_is_symmetric() {
    // Function applications are never evaluated at construction time, so
    // `Γ(zoo)` has unknown finiteness and `0 · Γ(zoo)` folds to `0` —
    // the same in both orders.
    let ctx = Context::new();
    let zero = ctx.int(0);
    let g = ctx.complex_infinity().gamma();
    assert_eq!(&zero * &g, &g * &zero);
    let e = ctx.neg_infinity().exp();
    assert_eq!(&zero * &e, &e * &zero);
}
