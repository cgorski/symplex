//! Limits that hung or were refused after seconds (0.44 → 0.45), found by
//! the 0.44 limit hunter and the comparison of 0.43 with 0.44.  Each test
//! is a reproducer with a time limit generous for a debug build on a loaded
//! machine (the times quoted are debug builds, CPU seconds), and its answer
//! is checked against the oracle cited: mpmath 1.3 at 40 digits along
//! `a ± 10⁻³⁰`, SymPy 1.14 as a second opinion.

use std::time::{Duration, Instant};

use symplex::prelude::*;

const B: Direction = Direction::Both;
const L: Direction = Direction::Left;
const R: Direction = Direction::Right;

/// `lim_{x→point} src` (direction `dir`) within `bound`, and its value
/// `re + i·im` to 12 digits.
fn value_within(src: &str, point: &str, dir: Direction, (re, im): (f64, f64), bound: Duration) {
    let ctx = Context::new();
    let f = ctx
        .parse(src)
        .unwrap_or_else(|e| panic!("`{src}` does not parse: {e}"));
    let x = ctx.symbol("x");
    let p = ctx.parse(point).unwrap();
    let t0 = Instant::now();
    let r = f.try_limit_dir(&x, &p, dir);
    let dt = t0.elapsed();
    assert!(
        dt < bound,
        "limit of `{src}` at {point} {dir:?} took {dt:?}"
    );
    let got = r.unwrap_or_else(|e| panic!("limit of `{src}` at {point} {dir:?} failed: {e}"));
    let z = got.eval_complex64().unwrap();
    let scale = re.hypot(im).max(1.0);
    assert!(
        (z.re - re).abs() < 1e-12 * scale && (z.im - im).abs() < 1e-12 * scale,
        "`{src}` at {point} {dir:?}: {got} = {z}, want {re} + {im}i"
    );
}

/// Before: more than 45 s each (hangs of the 0.44 limit hunter; 0.43 took
/// less than 0.5 s), now 0.3 s.  `x·(acosh(c + a·x) − acosh(c))⁻¹` is a
/// product of two factors tending to `0` and `∞`, undecided by the limits
/// of its factors; the limit of the second, `+∞`, was completed by its
/// imaginary part `Im(1/(acosh(c + ix) − acosh(c)))`, which vanishes
/// identically (`acosh(iy) = asinh(y) + iπ/2` for `y > 0`): the Gruntz
/// algorithm expanded the identically zero reflection `(e − ē)/(2i)` at
/// growing precision, its coefficients swelling with `√(2i ± 1)` (104 s for
/// that limit alone).  A product uses only the directions of its factors'
/// infinite limits.
///
/// Oracle: mpmath at `±10⁻³⁰`: `√5 = 2.2360679774997896964…` for `c = 2i`,
/// `a = i` from both sides, `−√5` for `c = −2i`, `√5/2` for `c = i/2`,
/// `√(1 + π²/4) = 1.8620958891185866250…` for `c = acosh 0 = iπ/2`, and the
/// opposite values for `a = −i`.  SymPy: `−i·√(−1 + 2i)·√(1 + 2i)` & c.
#[test]
fn acosh_quotients_on_the_imaginary_axis() {
    let bound = Duration::from_secs(10);
    let sqrt5 = 5f64.sqrt();
    let half_pi = std::f64::consts::FRAC_PI_2;
    let cases = [
        ("2*I", sqrt5),
        ("-2*I", -sqrt5),
        ("I/2", sqrt5 / 2.0),
        ("acosh(0)", (1.0 + half_pi * half_pi).sqrt()),
    ];
    for (c, v) in cases {
        for (a, s) in [("I", 1.0), ("-I", -1.0)] {
            for dir in [R, L] {
                let src = format!("x/(acosh({c} + ({a})*x) - acosh({c}))");
                value_within(&src, "0", dir, (s * v, 0.0), bound);
            }
        }
    }
}

/// `Γ(−2 + ix)·ix` and `Γ(−2 + ix)·x` at `0`: 1 ms in 0.43, 0.17 s in 0.44
/// (which finds `lim Γ(−2 + ix) = i·∞`, and searched for its real part
/// for the `0·∞` product), now 5 ms.
///
/// Oracle: the residue `1/2` of `Γ` at `−2`: `1/2` and `−i/2` (mpmath at
/// `±10⁻³⁰`: `0.5`, `−0.5i`).
#[test]
fn pole_factor_of_a_product() {
    let bound = Duration::from_secs(5);
    for dir in [R, L] {
        value_within("gamma(-2 + I*x)*(I*x)", "0", dir, (0.5, 0.0), bound);
        value_within("gamma(-2 + I*x)*x", "0", dir, (0.0, -0.5), bound);
    }
}

/// Before: refused after 3.5 s ("maximum recursion depth exceeded"), now
/// 0.3 s (two-sided), 0.01 s (from the left), 0.3 s (from the right).
/// `xˣ` is real near `x = π`, so `atanh(xˣ)` stays on the cut `(1, ∞)`
/// of `atanh`, where its principal values are continuous: the limits of
/// the composition, and the analytic fallback (L'Hôpital's rule), asked
/// for an argument real for every real `x` instead; the leading term of
/// the one-sided analysis expanded ten terms of `atanh(xˣ)` and `cos(xˣ)`
/// (constants in `π^π`) before a short expansion would have shown its
/// first non-zero coefficient; and `1/(cos(xˣ) − cos(π^π))` was refused,
/// as the sign of `ln π + 1` was unknown.
///
/// Oracle: mpmath at `π ± 10⁻²⁰`: `−0.00079672208760790455991…` from both
/// sides, the value of `1/((π^{2π} − 1)·sin(π^π))`; `±1.35·10⁸` for
/// `1/(cos(xˣ) − cos(π^π))` at `π ± 10⁻¹⁰` (`sin(π^π) = −0.9448 < 0`).
/// SymPy says the limit does not exist (its `atanh` written as logarithms
/// leaves the cut from the wrong side).
#[test]
fn atanh_of_a_power_tower_on_its_cut() {
    let bound = Duration::from_secs(10);
    let src = "(atanh(x^x) - atanh(pi^pi))/(cos(x^x) - cos(pi^pi))";
    let v = -0.000_796_722_087_607_904_6;
    for dir in [B, L, R] {
        value_within(src, "pi", dir, (v, 0.0), bound);
    }
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.parse("pi").unwrap();
    let f = ctx.parse("1/(cos(x^x) - cos(pi^pi))").unwrap();
    let t0 = Instant::now();
    let right = f.try_limit_dir(&x, &pi, R).unwrap();
    let left = f.try_limit_dir(&x, &pi, L).unwrap();
    assert!(t0.elapsed() < bound);
    assert_eq!(right, ctx.parse("oo").unwrap());
    assert_eq!(left, ctx.parse("-oo").unwrap());
}
