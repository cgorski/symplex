//! Limits where a complex argument meets a branch cut.

use symplex::prelude::*;

fn c64(e: &Ex) -> num_complex::Complex64 {
    e.eval_complex64().unwrap()
}

/// The nightly `fuzz_calculus` failure of 2026-10-03:
/// `limit(x/(acosh(acosh(x)) − acosh(acosh(0))), x, 0, '+')` was `0`.
/// Gruntz rewrote `acosh u` as `ln(u + √(u² − 1))`, which is `acosh u` only
/// for `Re u > 0`; at `u = acosh 0 = iπ/2` the radicand is a negative real
/// and the leading term vanished.  The rewrite is now SymPy's
/// `ln(u + √(u + 1)·√(u − 1))`, valid on all of ℂ.
///
/// SymPy: `limit(x/(acosh(acosh(x)) - acosh(acosh(0))), x, 0, '+')` →
/// `I*sqrt(-2 + I*pi)*sqrt(2 + I*pi)/2` ≈ −1.86209588911859; mpmath at
/// `x = 10⁻³⁰`: −1.862095889118586625.
#[test]
fn nested_acosh_at_its_cut() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let f = ctx.parse("x/(acosh(acosh(x)) - acosh(acosh(0)))").unwrap();
    for dir in [Direction::Right, Direction::Left, Direction::Both] {
        let l = f.try_limit_dir(&x, &zero, dir).unwrap();
        let z = c64(&l);
        assert!(
            (z.re + 1.862_095_889_118_586_6).abs() < 1e-12 && z.im.abs() < 1e-12,
            "{dir:?}: {l} = {z}"
        );
    }
}

/// `acosh x` at `−∞`: the old rewrite `ln(x + √(x² − 1))` is `−acosh(−x)`
/// there.  SymPy: `limit(acosh(x) - log(-x), x, -oo)` → `log(2) + I*pi`,
/// `limit(acosh(x)/log(-x), x, -oo)` → `1` (both were `−∞` and `−1`).
#[test]
fn acosh_at_minus_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let ninf = ctx.neg_infinity();
    let a = ctx
        .parse("acosh(x) - ln(-x)")
        .unwrap()
        .try_limit(&x, &ninf)
        .unwrap();
    assert_eq!(a.to_string(), "pi*I + ln(2)");
    let b = ctx
        .parse("acosh(x)/ln(-x)")
        .unwrap()
        .try_limit(&x, &ninf)
        .unwrap();
    assert_eq!(b.to_string(), "1");
}

/// A non-real argument crossing a branch cut at the point: the two one-sided
/// limits differ, and symplex gave one value for both sides (L'Hôpital's
/// rule assumes analyticity; the series engine took the Taylor coefficients
/// along the cut).  Now each side is right or refused.
///
/// mpmath at `x = ±10⁻²⁵` (SymPy `limit` agrees): `x/(ln(−2 + ix) − ln(−2))`
/// → `2i` from the right, `0` from the left (was `2i` both);
/// `x/(asin(2 + ix) − asin 2)` → `0`, `√3` (was `√3` both);
/// `x/(atan(2i + x) − atan(2i))` → `−3`, `0` (was `−3` both).
#[test]
fn one_sided_limits_across_a_cut() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    type Point = (f64, f64);
    let cases: [(&str, Point, Point); 3] = [
        ("x/(ln(-2 + I*x) - ln(-2))", (0.0, 2.0), (0.0, 0.0)),
        (
            "x/(asin(2 + I*x) - asin(2))",
            (0.0, 0.0),
            (3f64.sqrt(), 0.0),
        ),
        ("x/(atan(2*I + x) - atan(2*I))", (-3.0, 0.0), (0.0, 0.0)),
    ];
    for (src, right, left) in cases {
        let f = ctx.parse(src).unwrap();
        for (dir, want) in [(Direction::Right, right), (Direction::Left, left)] {
            if let Ok(l) = f.try_limit_dir(&x, &zero, dir) {
                let z = c64(&l);
                assert!(
                    (z.re - want.0).abs() < 1e-12 && (z.im - want.1).abs() < 1e-12,
                    "{src} {dir:?}: {l}, want {want:?}"
                );
            }
        }
        assert!(
            f.try_limit_dir(&x, &zero, Direction::Both).is_err(),
            "{src}"
        );
    }
}
