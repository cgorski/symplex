//! The assumption system (`is_positive`, `is_real`, `is_finite`, …) on
//! expressions that may be infinite, on the order of `polygamma`, and on
//! deep expressions.  An undeclared symbol is a finite complex number
//! (decision D4), so `|x|` stays real and `sinh x` finite; a symbol declared
//! `ExtendedReal` or `Infinite`, and an expression containing `oo`, `-oo` or
//! `zoo`, may be infinite.  Each test says what was answered before and
//! cites its oracle: an admissible point where the old claim fails, with
//! the value SymPy 1.14 gives there.

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `oo^(2x)` for a real `x` was positive and real; at `x = −1` it is
/// `oo^(−2) = 0` (SymPy `oo**(-2)` → `0`) and at `x = 1` it is `oo`.  For
/// `p` declared positive and infinite (`+oo`), `p²` was real and `1/p`
/// positive, but `oo**2` is `oo` and `1/oo` is `0`.  Now these powers are
/// only non-negative; finite positive bases keep their answers.
#[test]
fn powers_of_infinite_positive_bases_are_only_nonnegative() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let e = parse(&ctx, "oo**(2*x)");
    assert_eq!(e.is_positive(), None, "{e}");
    assert_eq!(e.is_real(), None, "{e}");
    assert_eq!(e.is_finite(), None, "{e}");
    assert_eq!(e.is_nonnegative(), Some(true), "{e}");
    assert_eq!(e.subs(&x, &ctx.int(-1)), ctx.int(0));

    let p = ctx
        .symbol_with("p", &[Assumption::Positive, Assumption::Infinite])
        .unwrap();
    for e in [p.powi(2), p.powi(-1), p.sqrt()] {
        assert_eq!(e.is_positive(), None, "{e}");
        assert_eq!(e.is_real(), None, "{e}");
        assert_eq!(e.is_nonnegative(), Some(true), "{e}");
    }

    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let e = a.pow(&x);
    assert_eq!((e.is_positive(), e.is_real()), (Some(true), Some(true)));
}

/// `u^oo` for a non-negative `u` was real; at `u = 2` it is `oo` (SymPy
/// `2**oo` → `oo`).  It is still non-negative (`0`, `oo`, or undefined).
#[test]
fn nonnegative_base_to_an_infinite_power_is_not_real() {
    let ctx = Context::new();
    let u = ctx.symbol_with("u", &[Assumption::NonNegative]).unwrap();
    let e = u.pow(&ctx.infinity());
    assert_eq!(e.is_real(), None, "{e}");
    assert_eq!(e.is_finite(), None, "{e}");
    assert_eq!(e.is_nonnegative(), Some(true), "{e}");
    let v = ctx.symbol_with("v", &[Assumption::Positive]).unwrap();
    assert_eq!(u.pow(&v).is_real(), Some(true));
}

/// `|oo·x|` and `|y|` for an `ExtendedReal` `y` were real (so finite); at
/// `x = 1` and `y = oo` both are `oo` (SymPy `Abs(oo)` → `oo`).  They are
/// non-negative extended reals; `|x|` of an undeclared `x` stays real.
#[test]
fn abs_of_a_possibly_infinite_value_is_not_real() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol_with("y", &[Assumption::ExtendedReal]).unwrap();
    for e in [(&x * &ctx.infinity()).abs(), y.abs()] {
        assert_eq!(e.is_real(), None, "{e}");
        assert_eq!(e.is_finite(), None, "{e}");
        assert_eq!(e.is_nonnegative(), Some(true), "{e}");
        assert_eq!(e.query(Props::EXTENDED_REAL), Some(true), "{e}");
    }
    assert_eq!(x.abs().is_real(), Some(true));
    assert_eq!(x.abs().is_finite(), Some(true));
}

/// `re y` and `im y` of an `Infinite` `y` were real; at `y = oo`,
/// `re(oo) = oo` (SymPy).  `re`, `im` of an undeclared symbol stay real.
#[test]
fn re_and_im_of_a_possibly_infinite_value_are_extended_real() {
    let ctx = Context::new();
    let y = ctx.symbol_with("y", &[Assumption::Infinite]).unwrap();
    for e in [y.re(), y.im()] {
        assert_eq!(e.is_real(), None, "{e}");
        assert_eq!(e.query(Props::EXTENDED_REAL), Some(true), "{e}");
    }
    let x = ctx.symbol("x");
    assert_eq!(x.re().is_real(), Some(true));
    assert_eq!(x.im().is_real(), Some(true));
}

/// `sinh y`, `cosh y`, `asinh y`, `acosh y`, `asin y`, `acos y`, `tanh y`,
/// `atan y`, `atanh y` were finite whatever `y`; for an `ExtendedReal` `y`
/// at `y = oo`: `sinh oo = cosh oo = asinh oo = acosh oo = oo`,
/// `asin oo = −oo·i`, `acos oo = oo·i` (SymPy).  They are finite for a
/// finite argument, so for an undeclared one.
#[test]
fn hyperbolic_and_inverse_trig_functions_of_infinite_arguments_are_not_finite() {
    let ctx = Context::new();
    let y = ctx.symbol_with("y", &[Assumption::ExtendedReal]).unwrap();
    let x = ctx.symbol("x");
    let fs: [fn(&Ex) -> Ex; 9] = [
        |e| e.sinh(),
        |e| e.cosh(),
        |e| e.tanh(),
        |e| e.asinh(),
        |e| e.acosh(),
        |e| e.atanh(),
        |e| e.asin(),
        |e| e.acos(),
        |e| e.atan(),
    ];
    for f in fs {
        let e = f(&y);
        assert_eq!(e.is_finite(), None, "{e}");
        let e = f(&x);
        assert_eq!(e.is_finite(), Some(true), "{e}");
    }
}

/// `ln p` and `ln Γ(p)` for `p` declared positive and infinite were real;
/// `ln oo = oo` and `loggamma(oo) = oo` (SymPy).  `ln p` is extended real.
#[test]
fn logarithms_of_an_infinite_positive_value_are_not_real() {
    let ctx = Context::new();
    let p = ctx
        .symbol_with("p", &[Assumption::Positive, Assumption::Infinite])
        .unwrap();
    let e = p.ln();
    assert_eq!(e.is_real(), None, "{e}");
    assert_eq!(e.query(Props::EXTENDED_REAL), Some(true), "{e}");
    let e = p.log_gamma();
    assert_eq!(e.is_real(), None, "{e}");
    let q = ctx.symbol_with("q", &[Assumption::Positive]).unwrap();
    assert_eq!(q.ln().is_real(), Some(true));
    assert_eq!(q.log_gamma().is_real(), Some(true));
}

/// `polygamma(n, x)` was real for every order `n` once `x > 0`; SymPy's
/// generalised polygamma has `polygamma(I, 2) = 1.0747… + 1.9380…·i`
/// (`polygamma(I, 2).evalf()`), so `polygamma(I, 2)` and
/// `polygamma(k, 2)` for an undeclared `k` were wrongly real.  A real order
/// keeps the answer (`polygamma(1/2, 2) = 0.8301…`, real).
#[test]
fn polygamma_is_real_only_for_a_real_order() {
    let ctx = Context::new();
    // `x.polygamma(&n)` is `polygamma(n, x)`.
    let two = ctx.int(2);
    let e = two.polygamma(&ctx.i_unit());
    assert_eq!(e.is_real(), None, "{e}");
    let k = ctx.symbol("k");
    let e = two.polygamma(&k);
    assert_eq!(e.is_real(), None, "{e}");
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    assert_eq!(x.polygamma(&ctx.int(1)).is_real(), Some(true));
    assert_eq!(x.polygamma(&ctx.rational(1, 2)).is_real(), Some(true));
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    assert_eq!(x.polygamma(&r).is_real(), Some(true));
}

/// Expressions at a pole for every allowed value were real (or complex)
/// and finite: `Γ(n)`, `ψ(n)`, `polygamma(2, n)` for a negative integer
/// `n` (SymPy `gamma(-2)`, `digamma(-2)`, `polygamma(2, -2)` → `zoo`),
/// `ln z`, `z⁻¹` and `Ei z` for `z` declared zero (`log(0)`, `1/0` →
/// `zoo`, `Ei(0)` → `-oo`).  Now undecided; `Γ(m)` for a positive integer
/// `m` stays real.
#[test]
fn functions_at_a_pole_everywhere_are_not_finite() {
    let ctx = Context::new();
    let n = ctx
        .symbol_with("n", &[Assumption::Integer, Assumption::Negative])
        .unwrap();
    let z = ctx.symbol_with("z", &[Assumption::Zero]).unwrap();
    for e in [
        n.gamma(),
        n.digamma(),
        n.polygamma(&ctx.int(2)),
        z.ln(),
        z.powi(-1),
        z.ei(),
    ] {
        assert_eq!(e.is_real(), None, "{e}");
        assert_eq!(e.is_complex(), None, "{e}");
        assert_eq!(e.is_finite(), None, "{e}");
        assert_eq!(e.is_rational(), None, "{e}");
    }
    let m = ctx
        .symbol_with("m", &[Assumption::Integer, Assumption::Positive])
        .unwrap();
    assert_eq!(m.gamma().is_real(), Some(true));
}

/// A query on `exp(exp(…exp(x)…))` 20,000 levels deep (and building it,
/// whose canonicalisation asks the assumption system) overflowed the stack
/// of the calling thread, which aborts the process: the assumption cache
/// recursed into the operands.  Now it is computed bottom-up.
#[test]
fn deep_expressions_do_not_overflow_the_stack() {
    let h = std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(|| {
            let ctx = Context::new();
            let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
            let mut e = x.clone();
            for _ in 0..20_000 {
                e = e.exp();
            }
            assert_eq!(e.is_positive(), Some(true));
            assert_eq!(e.is_real(), Some(true));
            let mut s = x.clone();
            for _ in 0..20_000 {
                s = s.sinh();
            }
            assert_eq!(s.is_real(), Some(true));
            assert_eq!(s.is_finite(), Some(true));
            assert_eq!(s.refine(), s);
        })
        .unwrap();
    h.join().unwrap();
}
