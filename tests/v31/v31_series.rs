//! Series at the branch points of the inverse trigonometric and hyperbolic
//! functions.

use std::time::{Duration, Instant};
use symplex::prelude::*;

/// The nightly `fuzz_calculus` input of 2026-09-28: `series(asin(x²), x,
/// 1, 4)` has no two-sided expansion (a square-root branch point), and it
/// took 6.5 s to say so (over 30 s on CI); `asin(x³)` took 42 s.  The
/// differentiation fallback asked the limit engine for the limits of ever
/// larger derivatives.  Now the branch point is recognised and the refusal
/// is immediate.
///
/// SymPy: `series(asin(x**2), x, 1, 4)` → `pi/2 - 2*I*sqrt(x - 1) - …`
/// (a Puiseux series, which is not a two-sided expansion over the reals).
#[test]
fn two_sided_series_at_a_branch_point_is_refused_at_once() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (src, point) in [
        ("asin(x*x)", 1),
        ("asin(x^3)", 1),
        ("asin(x^3)", -1),
        ("acos(x^2)", -1),
        ("atanh(x^2)", 1),
        ("acosh(x^2)", 1),
        ("acosh(x^3)", -1),
        ("asinh(I*x^2)", 1),
        ("atan(I*x^3)", 1),
    ] {
        let e = ctx.parse(src).unwrap();
        let t = Instant::now();
        let r = e.try_series(&x, &ctx.int(point), 4);
        assert!(r.is_err(), "{src} at {point}: {r:?}");
        assert!(
            t.elapsed() < Duration::from_secs(2),
            "{src} at {point} took {:?}",
            t.elapsed()
        );
    }
}

/// Where the argument meets the branch point to an order that makes the
/// square root a power series, the two-sided expansion exists and is exact
/// (it used to be refused after the same slow fallback).
///
/// SymPy: `series(asin(1 - x**4), x, 0, 7)` →
/// `pi/2 - sqrt(2)*x**2 - sqrt(2)*x**6/12 + O(x**7)`.
#[test]
fn branch_point_met_to_even_order_expands_two_sided() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = (1 - &x.powi(4)).asin().series(&x, &ctx.int(0), 7);
    assert_eq!(s.to_string(), "-1/12*sqrt(2)*x^6 - sqrt(2)*x^2 + 1/2*pi");
    // asin(1 − x²) meets it to order 2: √(2x²) = √2·|x|, no expansion.
    assert!(
        (1 - &x.powi(2))
            .asin()
            .try_series(&x, &ctx.int(0), 4)
            .is_err()
    );
}

/// One-sided expansions at a branch point are Puiseux series, as SymPy's
/// (SymPy's default direction is `+`).
///
/// SymPy: `series(asin(x**2), x, 1, 4)` → `pi/2 - 2*I*sqrt(x - 1) -
/// I*(x - 1)**(3/2)/6 + 13*I*(x - 1)**(5/2)/80 - 37*I*(x - 1)**(7/2)/448`;
/// `series(sqrt(x**2 - 1), x, 1, 3)` → `sqrt(2)*sqrt(x - 1) +
/// sqrt(2)*(x - 1)**(3/2)/4 - sqrt(2)*(x - 1)**(5/2)/32`;
/// `series(atanh(x**2), x, 1, 3)` → `-I*pi/2 - 1/4 - log(x - 1)/2 +
/// (x - 1)**2/16 + x/4`;
/// `series(acosh(x**2), x, 1, 2, dir='-')` → `2*I*sqrt(1 - x) -
/// I*(1 - x)**(3/2)/6`.
#[test]
fn one_sided_series_at_a_branch_point_is_puiseux() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let a = x.powi(2).asin().series_dir(&x, &one, 4, Direction::Right);
    assert_eq!(
        a.to_string(),
        "-1/6*(x - 1)^(3/2)*I - 2*sqrt(x - 1)*I - 37/448*(x - 1)^(7/2)*I + \
         13/80*(x - 1)^(5/2)*I + 1/2*pi"
    );
    let r = (&x.powi(2) - 1)
        .sqrt()
        .series_dir(&x, &one, 3, Direction::Right);
    assert_eq!(
        r.to_string(),
        "-1/32*sqrt(2)*(x - 1)^(5/2) + 1/4*sqrt(2)*(x - 1)^(3/2) + sqrt(2)*sqrt(x - 1)"
    );
    let h = x.powi(2).atanh().series_dir(&x, &one, 3, Direction::Right);
    assert_eq!(
        h.to_string(),
        "1/4*x + 1/16*(x - 1)^2 - 1/2*ln(x - 1) - 1/2*pi*I - 1/4"
    );
    let c = x.powi(2).acosh().series_dir(&x, &one, 2, Direction::Left);
    assert_eq!(c.to_string(), "-1/6*(-x + 1)^(3/2)*I + 2*sqrt(-x + 1)*I");
    // From the left asin(x²) is real: π/2 − 2√(1 − x) + …, and the series
    // agrees with the function next to the point.
    let l = x.powi(2).asin().series_dir(&x, &one, 3, Direction::Left);
    let xv = ctx.rational(999_999, 1_000_000);
    let want = x.powi(2).asin().subs(&x, &xv).eval_f64().unwrap();
    let got = l.subs(&x, &xv).eval_f64().unwrap();
    assert!((want - got).abs() < 1e-14, "{want} vs {got}: {l}");
}

/// A branch point hidden in a constant that is not written `1`:
/// `asin(sin 1) = 1`, so `asin(asin(x))` at `x = sin 1` has one; the
/// fallback took 6 s per side before refusing.
#[test]
fn hidden_branch_point_is_recognised() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.asin().asin();
    let point = ctx.int(1).sin();
    let t = Instant::now();
    assert!(e.try_series(&x, &point, 3).is_err());
    let left = e.series_dir(&x, &point, 2, Direction::Left);
    assert!(!left.has_unevaluated(), "{left}");
    assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
    let xv = &point - &ctx.rational(1, 10_000_000);
    let want = e.subs(&x, &xv).eval_f64().unwrap();
    let got = left.subs(&x, &xv).eval_f64().unwrap();
    assert!((want - got).abs() < 1e-9, "{want} vs {got}: {left}");
}
