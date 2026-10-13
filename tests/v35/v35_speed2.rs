//! Speed regressions since 0.43 that 0.45 still had, found by replaying the
//! fuzz corpora per unit in 0.43, 0.44 and 0.45 (CPU of each unit, then
//! `-runs=3` rechecks of the units at least twice and 50 ms slower).  Each
//! test is a reproducer with a time limit generous for a debug build on a
//! loaded machine, and its answer is checked (by differentiation, by
//! substitution, or against the oracle cited).  The times quoted are release
//! builds, CPU of the operation alone: 0.43, 0.45, and now.

use std::time::{Duration, Instant};

use symplex::prelude::*;

/// `∫ src dx` within `limit`, its answer checked by differentiation (or
/// unevaluated, when `closed` is false).
fn integrate_within(src: &str, limit: Duration, closed: bool) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse(src).unwrap();
    let t0 = Instant::now();
    let big_f = f.integrate(&x);
    let dt = t0.elapsed();
    assert!(dt < limit, "∫ {src} took {dt:?}");
    if !closed {
        assert!(big_f.has_unevaluated(), "∫ {src} = {big_f}");
        return;
    }
    assert!(!big_f.has_unevaluated(), "∫ {src} = {big_f}");
    let residual = &big_f.diff(&x) - &f;
    let mut checked = 0;
    for point in ["1/3", "7/5", "13/11", "5/2"] {
        let point = ctx.parse(point).unwrap();
        let (Ok(r), Ok(s)) = (
            residual.subs(&x, &point).eval_complex64(),
            f.subs(&x, &point).eval_complex64(),
        ) else {
            continue;
        };
        assert!(
            r.norm() < 1e-9 * s.norm().max(1.0),
            "∫ {src} = {big_f}: F′ − f = {r} at {point}"
        );
        checked += 1;
    }
    assert!(checked >= 2, "∫ {src} = {big_f}: too few sample points");
}

/// Before: 39, 45 and 67 ms (1, 1 and 2 ms in 0.43), now 11, 9 and 14 ms.
/// The logarithmic substitution `x = eʷ` of 0.44 hands the Risch tower
/// `R(w, e^{w/2})`, and the tower's Rothstein–Trager step searched the
/// resultant `R(z)` for roots among some 1,600 small rationals, each an
/// evaluation of `R` over `ℚ(w)` (4 ms per call, several calls per
/// integrand).  The tower reads only whether a root is non-constant, which
/// a resultant of degree two or more never shows there; the search is no
/// longer run.  The answers are unchanged: unevaluated, and `ln|x + ln x|`
/// for `(x + 1)/(x·(x + ln x))`, a closed form the substitution found in
/// 0.45 (unevaluated in 0.43).
#[test]
fn tower_rothstein_trager_reads_only_its_verdict() {
    // Debug build: 0.6 s for all four.
    let bound = Duration::from_secs(2);
    for src in [
        "(x + 1)/((x + sqrt(x))*ln(x))",
        "(-x + sqrt(x))/(x^3 + ln(x))",
        "(x - 3)/((-x - 1)*(sqrt(x) - ln(x)))",
    ] {
        integrate_within(src, bound, false);
    }
    integrate_within("(x + 1)/(x*(x + ln(x)))", bound, true);
}

/// `solve` over a number field (0.45, Trager) for `(x − ∛2)·(x − 1 + i)²`
/// expanded: 0.21 s, now 7 ms (0.43 refused: "roots not found in closed
/// form"); `x·(x − √2)·(x − ∛3)` expanded: 0.23 s, now 10 ms.  Each atom
/// (`∛2`, `i`) was placed in `K = ℚ(∛2 + i)` by an exact zero test of
/// `β − g(α)` for every root `g(α)` of its minimal polynomial over `K`: a
/// minimal polynomial of degree up to 18 factored over ℤ, 70 ms each.  For
/// `α` a sum of the atoms whose degree is the product of theirs, `g(α) = β`
/// is an identity in `ℚ[u, v]/(u³ − 2, v² + 1)`; other roots face a
/// certified evaluation first, and only a difference that is zero to its
/// precision takes the exact test.  Answers checked by substitution.
#[test]
fn number_field_atoms_located_without_zero_tests() {
    // Debug build: 0.15 s for both (0.45: 0.22 s each in a release build).
    let bound = Duration::from_secs(2);
    let cases = [
        (
            "x^3 + (-2 + 2*I - cbrt(2))*x^2 + (-2*I + 2*cbrt(2) - 2*I*cbrt(2))*x + 2*I*cbrt(2)",
            ["cbrt(2)", "1 - I"].as_slice(),
        ),
        (
            "x^3 - (sqrt(2) + cbrt(3))*x^2 + sqrt(2)*cbrt(3)*x",
            ["0", "sqrt(2)", "cbrt(3)"].as_slice(),
        ),
    ];
    for (src, want) in cases {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let f = ctx.parse(src).unwrap();
        let t0 = Instant::now();
        let roots = f.solve(&x).unwrap_or_else(|e| panic!("solve {src}: {e}"));
        let dt = t0.elapsed();
        assert!(dt < bound, "solve {src} took {dt:?}");
        assert_eq!(roots.len(), want.len(), "solve {src}: {roots:?}");
        for w in want {
            let w = ctx.parse(w).unwrap();
            assert!(
                roots.iter().any(|r| (r - &w).simplify() == ctx.int(0)),
                "solve {src}: {roots:?} lacks {w}"
            );
        }
    }
}

/// Parser fuzz inputs whose products with `zoo` (`0^(−70)`) asked whether
/// the other factors vanish: `((i!)!)!…` (38 nested factorials of complex
/// numbers) and `33^((i!)!…)` were evaluated, each `Γ` at rising precision
/// once complex `Γ` evaluated (0.44): 0.15 s and 0.05 s, now under 1 ms
/// (0.43: 3 ms and 0.4 ms, refusing the evaluation at once).  `Γ` has no
/// zeros, nor have `eᶻ` and `bᶻ` for a non-zero `b`.  Answer: `zoo`, as
/// before.
#[test]
fn factorials_are_never_zero() {
    // Debug build: under 10 ms for both.
    let bound = Duration::from_secs(1);
    for src in [
        "64!!!!!!!!!!!!4!!!!!7!!!i!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!5!!!!7!!!i!!!!!!!!!!!!!!!!!!!!!0**-70",
        "v4!!!!!!!!!!!!!!!!!!!!!!!33**i!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!33**i/0a-9!v",
    ] {
        let ctx = Context::new();
        let t0 = Instant::now();
        let e = symplex::parse::parse(&ctx, src).unwrap();
        let dt = t0.elapsed();
        assert!(dt < bound, "parse {src} took {dt:?}");
        assert_eq!(e, ctx.parse("zoo").unwrap(), "{src}");
    }
}

/// `acosh(i·v) = asinh|v| + i·sgn(v)·π/2` for a real `v ≠ 0` (mpmath 1.3:
/// `acosh(2i) = 1.44363547517881 + 1.5707963267949i`, `acosh(−2i)` its
/// conjugate, the same identity to 30 digits for `|v|` from `10⁻⁸` to
/// `10⁸`); up to 0.45 its parts stayed opaque.  The limit of `1/(acosh(2i +
/// ix) − acosh(2i))` at `0⁺` took 12.6 s to find `∞` (104 s in a debug
/// build): its Gruntz expansion of the imaginary part, which is
/// identically 0, swelled; now 23 ms (0.43: unevaluated, 18 ms).
#[test]
fn acosh_on_the_imaginary_axis() {
    let ctx = Context::new();
    for (src, want) in [
        ("re(acosh(2*I))", "asinh(2)"),
        ("im(acosh(2*I))", "pi/2"),
        ("re(acosh(-I/2))", "asinh(1/2)"),
        ("im(acosh(-I/2))", "-pi/2"),
        ("re(acosh(I*(1 - sqrt(3))))", "asinh(sqrt(3) - 1)"),
        ("im(acosh(I*(1 - sqrt(3))))", "-pi/2"),
    ] {
        assert_eq!(ctx.parse(src).unwrap(), ctx.parse(want).unwrap(), "{src}");
    }
    // Opaque off the axis and for a symbol of unknown sign.
    let x = ctx.symbol("x");
    let z = ctx.parse("re(acosh(I*x))").unwrap();
    assert!(format!("{z}").contains("re("), "{z}");
    let f = ctx.parse("1/(acosh(2*I + I*x) - acosh(2*I))").unwrap();
    let t0 = Instant::now();
    let r = f.try_limit_dir(&x, &ctx.int(0), Direction::Right);
    let dt = t0.elapsed();
    // Debug build: 0.25 s for the test (0.45: 104 s for this limit).
    assert!(dt < Duration::from_secs(5), "limit took {dt:?}");
    assert_eq!(r.unwrap(), ctx.parse("oo").unwrap());
}

/// `eval` of `acosh` of a nested complex power took ~1 s per call in the
/// 0.47 development (release; 2 ms in 0.46): the realness test ran the
/// exact cyclotomic walk over the radicals before looking at the value,
/// whose imaginary part (`10⁻¹³` of the modulus) a certified evaluation
/// shows at once.  `fuzz_simplify` timed out on
/// `acosh(((x^(-3/4))^(-3/2))^(-3/2))` (60 s for its sample points).
#[test]
fn realness_of_a_visibly_complex_value_is_decided_numerically() {
    let ctx = Context::new();
    let e = ctx
        .parse("acosh((((1/1000000000000*I + 13/4)^(-3/4))^(-3/2))^(-3/2))")
        .unwrap();
    let t = std::time::Instant::now();
    let v = e.eval();
    assert!(t.elapsed().as_secs_f64() < 2.0, "{:?}", t.elapsed());
    let z = v.eval_complex64().unwrap();
    // mpmath: acosh(((mpc(13/4, 1e-12)**(-0.75))**(-1.5))**(-1.5))
    assert!((z.im + 1.433_530_793_143_204_9).abs() < 1e-12, "{z}");
}
