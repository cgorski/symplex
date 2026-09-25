//! symplex 0.2 base-layer fixes — assumption lattice consistency for the
//! special constants.
//!
//! Model (SymPy's): `oo` is positive, extended_real, infinite, nonzero;
//! NOT finite, NOT real, NOT complex.  `positive` alone therefore only
//! implies `nonnegative ∧ nonzero ∧ extended_real`; `real` needs
//! `positive ∧ finite` (or `extended_real ∧ finite`).  No constant may be
//! self-contradictory after forward chaining, and every query must return
//! the same answer regardless of which property was asked first.

use symplex::prelude::*;

fn fresh() -> Context {
    Context::new()
}

/// Query `prop` on a *fresh* context so no earlier query can have primed
/// the cache, and also on a context that was primed with `first`.
fn q_fresh(build: fn(&Context) -> Ex, prop: Props) -> Option<bool> {
    let ctx = fresh();
    let e = build(&ctx);
    ctx.query(&e, prop)
}

fn q_primed(build: fn(&Context) -> Ex, first: Props, prop: Props) -> Option<bool> {
    let ctx = fresh();
    let e = build(&ctx);
    let _ = ctx.query(&e, first);
    ctx.query(&e, prop)
}

const ALL: [Props; 24] = [
    Props::COMMUTATIVE,
    Props::COMPLEX,
    Props::REAL,
    Props::RATIONAL,
    Props::INTEGER,
    Props::ALGEBRAIC,
    Props::TRANSCENDENTAL,
    Props::IRRATIONAL,
    Props::IMAGINARY,
    Props::POSITIVE,
    Props::NEGATIVE,
    Props::NONNEGATIVE,
    Props::NONPOSITIVE,
    Props::ZERO,
    Props::NONZERO,
    Props::EVEN,
    Props::ODD,
    Props::PRIME,
    Props::COMPOSITE,
    Props::FINITE,
    Props::INFINITE,
    Props::HERMITIAN,
    Props::ANTIHERMITIAN,
    Props::EXTENDED_REAL,
];

/// Every (first, prop) query order must agree with the fresh answer.
fn assert_order_independent(build: fn(&Context) -> Ex, label: &str) {
    for &prop in &ALL {
        let base = q_fresh(build, prop);
        for &first in &ALL {
            let got = q_primed(build, first, prop);
            assert_eq!(
                got, base,
                "{label}: query({prop}) = {base:?} fresh but {got:?} after query({first})"
            );
        }
    }
}

#[test]
fn oo_is_positive_infinite_extended_real_not_finite_not_real() {
    let ctx = fresh();
    let oo = ctx.infinity();
    assert_eq!(ctx.query(&oo, Props::POSITIVE), Some(true));
    assert_eq!(ctx.query(&oo, Props::NONNEGATIVE), Some(true));
    assert_eq!(ctx.query(&oo, Props::NONZERO), Some(true));
    assert_eq!(ctx.query(&oo, Props::EXTENDED_REAL), Some(true));
    assert_eq!(ctx.query(&oo, Props::INFINITE), Some(true));
    assert_eq!(ctx.query(&oo, Props::FINITE), Some(false));
    assert_eq!(ctx.query(&oo, Props::REAL), Some(false));
    assert_eq!(ctx.query(&oo, Props::COMPLEX), Some(false));
    assert_eq!(ctx.query(&oo, Props::ZERO), Some(false));
    assert_eq!(ctx.query(&oo, Props::NEGATIVE), Some(false));
    assert_eq!(ctx.query(&oo, Props::INTEGER), Some(false));
    assert_eq!(ctx.query(&oo, Props::IMAGINARY), Some(false));
}

#[test]
fn neg_oo_is_negative_infinite_extended_real_not_finite_not_real() {
    let ctx = fresh();
    let m = ctx.neg_infinity();
    assert_eq!(ctx.query(&m, Props::NEGATIVE), Some(true));
    assert_eq!(ctx.query(&m, Props::NONPOSITIVE), Some(true));
    assert_eq!(ctx.query(&m, Props::NONZERO), Some(true));
    assert_eq!(ctx.query(&m, Props::EXTENDED_REAL), Some(true));
    assert_eq!(ctx.query(&m, Props::INFINITE), Some(true));
    assert_eq!(ctx.query(&m, Props::FINITE), Some(false));
    assert_eq!(ctx.query(&m, Props::REAL), Some(false));
    assert_eq!(ctx.query(&m, Props::COMPLEX), Some(false));
    assert_eq!(ctx.query(&m, Props::POSITIVE), Some(false));
}

#[test]
fn zoo_is_infinite_and_nothing_real() {
    let ctx = fresh();
    let z = ctx.complex_infinity();
    assert_eq!(ctx.query(&z, Props::INFINITE), Some(true));
    assert_eq!(ctx.query(&z, Props::NONZERO), Some(true));
    assert_eq!(ctx.query(&z, Props::FINITE), Some(false));
    assert_eq!(ctx.query(&z, Props::REAL), Some(false));
    assert_eq!(ctx.query(&z, Props::EXTENDED_REAL), Some(false));
    assert_eq!(ctx.query(&z, Props::COMPLEX), Some(false));
    assert_eq!(ctx.query(&z, Props::POSITIVE), Some(false));
    assert_eq!(ctx.query(&z, Props::NEGATIVE), Some(false));
    assert_eq!(ctx.query(&z, Props::ZERO), Some(false));
}

#[test]
fn nan_knows_nothing_numeric() {
    let ctx = fresh();
    let n = ctx.nan();
    for &p in &ALL {
        if p == Props::COMMUTATIVE {
            assert_eq!(ctx.query(&n, p), Some(true));
        } else {
            assert_eq!(ctx.query(&n, p), None, "nan should not decide {p}");
        }
    }
}

#[test]
fn imaginary_unit_is_imaginary_finite_not_real() {
    let ctx = fresh();
    let i = ctx.i_unit();
    assert_eq!(ctx.query(&i, Props::IMAGINARY), Some(true));
    assert_eq!(ctx.query(&i, Props::COMPLEX), Some(true));
    assert_eq!(ctx.query(&i, Props::FINITE), Some(true));
    assert_eq!(ctx.query(&i, Props::ALGEBRAIC), Some(true));
    assert_eq!(ctx.query(&i, Props::NONZERO), Some(true));
    assert_eq!(ctx.query(&i, Props::REAL), Some(false));
    assert_eq!(ctx.query(&i, Props::EXTENDED_REAL), Some(false));
    assert_eq!(ctx.query(&i, Props::POSITIVE), Some(false));
    assert_eq!(ctx.query(&i, Props::NEGATIVE), Some(false));
    assert_eq!(ctx.query(&i, Props::ZERO), Some(false));
    assert_eq!(ctx.query(&i, Props::INFINITE), Some(false));
}

#[test]
fn constants_are_query_order_independent() {
    assert_order_independent(|c| c.infinity(), "oo");
    assert_order_independent(|c| c.neg_infinity(), "-oo");
    assert_order_independent(|c| c.complex_infinity(), "zoo");
    assert_order_independent(|c| c.nan(), "nan");
    assert_order_independent(|c| c.i_unit(), "I");
    assert_order_independent(|c| c.pi(), "pi");
    assert_order_independent(|c| c.e(), "E");
}

#[test]
fn no_constant_is_contradictory_after_chaining() {
    // A contradictory set answers Some(true) *and* would answer Some(false)
    // for the same property; detect that by asking for both a property and
    // its logical complement.
    let ctx = fresh();
    let pairs = [
        (Props::FINITE, Props::INFINITE),
        (Props::ZERO, Props::NONZERO),
        (Props::POSITIVE, Props::NONPOSITIVE),
        (Props::NEGATIVE, Props::NONNEGATIVE),
    ];
    for e in [
        ctx.infinity(),
        ctx.neg_infinity(),
        ctx.complex_infinity(),
        ctx.nan(),
        ctx.i_unit(),
    ] {
        for &(a, b) in &pairs {
            let qa = ctx.query(&e, a);
            let qb = ctx.query(&e, b);
            assert!(
                !(qa == Some(true) && qb == Some(true)),
                "{e}: {a} and {b} both true"
            );
        }
    }
}

#[test]
fn declared_positive_symbol_is_finite_and_real() {
    // As in SymPy, `Symbol('x', positive=True).is_real` is True: a symbol
    // *declared* positive is a finite positive number.  The lattice itself
    // must not derive `finite` from `positive` alone (otherwise `oo` would
    // be contradictory), so the declaration path adds `finite` explicitly.
    let ctx = fresh();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    assert_eq!(ctx.query(&x, Props::POSITIVE), Some(true));
    assert_eq!(ctx.query(&x, Props::NONNEGATIVE), Some(true));
    assert_eq!(ctx.query(&x, Props::NONZERO), Some(true));
    assert_eq!(ctx.query(&x, Props::EXTENDED_REAL), Some(true));
    assert_eq!(ctx.query(&x, Props::REAL), Some(true));
    assert_eq!(ctx.query(&x, Props::FINITE), Some(true));
    assert_eq!(ctx.query(&x, Props::COMPLEX), Some(true));
    assert_eq!(ctx.query(&x, Props::NEGATIVE), Some(false));
    assert_eq!(ctx.query(&x, Props::ZERO), Some(false));
}

#[test]
fn positive_and_infinite_is_consistent_in_the_lattice() {
    let mut a = Assumptions::default();
    a.assert_true(Props::POSITIVE);
    a.assert_true(Props::INFINITE);
    assert!(!a.is_contradictory(), "{a}");
    assert_eq!(a.query(Props::EXTENDED_REAL), Some(true));
    assert_eq!(a.query(Props::NONNEGATIVE), Some(true));
    assert_eq!(a.query(Props::NONZERO), Some(true));
    assert_eq!(a.query(Props::FINITE), Some(false));
    assert_eq!(a.query(Props::REAL), Some(false));

    let mut b = Assumptions::default();
    b.assert_true(Props::POSITIVE);
    b.assert_true(Props::FINITE);
    assert!(!b.is_contradictory(), "{b}");
    assert_eq!(b.query(Props::REAL), Some(true));
    assert_eq!(b.query(Props::COMPLEX), Some(true));
}

#[test]
fn declared_extended_real_symbol_stays_agnostic_about_finiteness() {
    let ctx = fresh();
    let e = ctx.symbol_with("e", &[Assumption::ExtendedReal]).unwrap();
    assert_eq!(ctx.query(&e, Props::EXTENDED_REAL), Some(true));
    assert_eq!(ctx.query(&e, Props::REAL), None);
    assert_eq!(ctx.query(&e, Props::FINITE), None);
    // An explicit Infinite is respected: `[Positive, Infinite]` is `+oo`.
    let p = ctx
        .symbol_with("p", &[Assumption::Positive, Assumption::Infinite])
        .unwrap();
    assert_eq!(ctx.query(&p, Props::POSITIVE), Some(true));
    assert_eq!(ctx.query(&p, Props::FINITE), Some(false));
    assert_eq!(ctx.query(&p, Props::REAL), Some(false));
    // NotReal symbols are finite complex numbers by default: not positive.
    let z = ctx.symbol_with("z", &[Assumption::NotReal]).unwrap();
    assert_eq!(ctx.query(&z, Props::POSITIVE), Some(false));
    assert_eq!(ctx.query(&z, Props::FINITE), Some(true));
}

#[test]
fn assume_builder_and_refine_with_follow_the_same_convention() {
    let ctx = fresh();
    let t = ctx.symbol("t").assume(Assumption::Positive).unwrap();
    assert_eq!(t.is_real(), Some(true));
    assert_eq!(t.is_finite(), Some(true));
    let x = ctx.symbol("x");
    // refine_with temporarily declares x positive → sqrt(x²) → x
    let r = x
        .powi(2)
        .sqrt()
        .refine_with(&[(&x, Assumption::Positive)])
        .unwrap();
    assert_eq!(r, x);
}

#[test]
fn contradictory_symbol_assumptions_are_rejected() {
    let ctx = fresh();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.symbol_with("bad", &[Assumption::Positive, Assumption::Negative])
            .unwrap()
    }));
    assert!(r.is_err(), "positive ∧ negative must be rejected");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.symbol_with("bad2", &[Assumption::Integer, Assumption::Irrational])
            .unwrap()
    }));
    assert!(r.is_err(), "integer ∧ irrational must be rejected");
}
