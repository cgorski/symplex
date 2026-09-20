//! 0.11.1 review fixes: `calculus_util` (monotonicity across poles, `abs`
//! kinks, error variants, honest docs) and the algebraic bridge
//! (`minimal_polynomial` verification, `RootOf` indexing, lock discipline).
//!
//! Reference values cite SymPy 1.14 (`symplex/.venv/bin/python`).  Where
//! SymPy tests the derivative alone and answers wrongly across a pole,
//! or raises `NotImplementedError` on `Abs`, its answer is quoted and the
//! mathematics is followed.

use std::thread;

use symplex::prelude::*;
use symplex::sym;

fn s<T: std::fmt::Display>(e: &T) -> String {
    format!("{e}")
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Monotonicity ignores poles inside the domain
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn monotonicity_is_refuted_by_a_pole_inside_the_domain() {
    let ctx = Context::new();
    sym!(ctx; x, Real);
    let m1 = ctx.interval(&ctx.int(-1), &ctx.int(1), false, false);
    let inv = ctx.int(1) / &x;

    // f(-1) = -1 < 1 = f(1): 1/x is not decreasing on [-1, 1], although
    // f' = -1/x² < 0 wherever it is defined.
    // SymPy: is_decreasing(1/x, Interval(-1, 1), x) is None (undecided);
    //        is_increasing(1/x, Interval(-1, 1), x) is False;
    //        is_monotonic(1/x, Interval(-1, 1), x) is True  (wrong: it only
    //        asks whether f' has zeros);
    //        is_strictly_decreasing(1/x, Interval(-1, 1), x) is None.
    assert_eq!(inv.is_decreasing(&x, &m1), Some(false));
    assert_eq!(inv.is_increasing(&x, &m1), Some(false));
    assert_eq!(inv.is_monotonic(&x, &m1), Some(false));
    assert_eq!(inv.is_strictly_decreasing(&x, &m1), Some(false));
    // -1/x has f' = 1/x² > 0 on both sides of the pole and is still not
    // increasing across it.  SymPy: is_increasing(-1/x, Interval(-1, 1), x)
    // is None.
    assert_eq!((-&inv).is_increasing(&x, &m1), Some(false));

    // tan on [0, π] jumps from +∞ to -∞ at π/2.
    // SymPy: is_increasing(tan(x), Interval(0, pi), x) is True (wrong),
    //        is_strictly_increasing(tan(x), Interval(0, pi), x) is True (wrong).
    let zero_pi = ctx.interval(&ctx.int(0), &ctx.pi(), false, false);
    assert_eq!(x.tan().is_increasing(&x, &zero_pi), Some(false));
    assert_eq!(x.tan().is_strictly_increasing(&x, &zero_pi), Some(false));
    // SymPy: is_decreasing(cot(x), Interval(-1, 1), x) is True (wrong: pole at 0).
    assert_eq!(x.cot().is_decreasing(&x, &m1), Some(false));
    // ln(x²) → -∞ at 0.  SymPy: is_increasing(log(x**2), Interval(-1, 1), x)
    // is False, is_decreasing(...) is False.
    assert_eq!(x.powi(2).ln().is_increasing(&x, &m1), Some(false));
    assert_eq!(x.powi(2).ln().is_decreasing(&x, &m1), Some(false));

    // Away from the pole the old answers stand.
    // SymPy: is_increasing(tan(x), Interval.open(-pi/2, pi/2), x) is True
    let branch = ctx.interval(&(-&ctx.pi() / 2), &(&ctx.pi() / 2), true, true);
    assert_eq!(x.tan().is_increasing(&x, &branch), Some(true));
    // SymPy: is_decreasing(1/x, Interval(1, 2), x) is True; is_increasing is False
    let one_two = ctx.interval(&ctx.int(1), &ctx.int(2), false, false);
    assert_eq!(inv.is_decreasing(&x, &one_two), Some(true));
    assert_eq!(inv.is_increasing(&x, &one_two), Some(false));
    // SymPy: is_increasing(x**3, Interval(-1, 1), x) is True
    assert_eq!(x.powi(3).is_increasing(&x, &m1), Some(true));

    // Infinitely many poles that cannot be enumerated: undecided, never
    // a guess.  SymPy: is_increasing(tan(x), S.Reals, x) is True (wrong).
    assert_ne!(x.tan().is_increasing(&x, &ctx.reals()), Some(true));
}

#[test]
fn convexity_inherits_the_pole_check() {
    let ctx = Context::new();
    sym!(ctx; x, Real);
    let m1 = ctx.interval(&ctx.int(-1), &ctx.int(1), false, false);
    let zero_pi = ctx.interval(&ctx.int(0), &ctx.pi(), false, false);
    // SymPy: is_convex(1/x, x, domain=Interval(-1, 1)) is False;
    //        is_convex(tan(x), x, domain=Interval(0, pi)) is False
    assert_eq!((ctx.int(1) / &x).is_convex(&x, &m1), Some(false));
    assert_eq!(x.tan().is_convex(&x, &zero_pi), Some(false));
    // SymPy: is_convex(1/x, x, domain=Interval.open(0, oo)) is True
    let pos = ctx.interval(&ctx.int(0), &ctx.infinity(), true, true);
    assert_eq!((ctx.int(1) / &x).is_convex(&x, &pos), Some(true));
}

#[test]
fn monotonicity_with_real_and_positive_symbols() {
    let ctx = Context::new();
    sym!(ctx; p, Positive);
    let pos = ctx.interval(&ctx.int(0), &ctx.infinity(), true, true);
    let half = ctx.interval(&ctx.int(0), &ctx.infinity(), false, true);

    // SymPy (p positive): is_increasing(sqrt(p), Interval.open(0, oo), p) is True,
    // is_strictly_increasing(...) is True
    assert_eq!(p.sqrt().is_increasing(&p, &pos), Some(true));
    assert_eq!(p.sqrt().is_strictly_increasing(&p, &pos), Some(true));
    // SymPy: is_decreasing(exp(-p), Interval(0, oo), p) is True
    assert_eq!((-&p).exp().is_decreasing(&p, &half), Some(true));
    // SymPy: is_increasing(log(p), Interval.open(0, oo), p) is True,
    //        is_decreasing(log(p), Interval.open(0, oo), p) is False
    assert_eq!(p.ln().is_increasing(&p, &pos), Some(true));
    assert_eq!(p.ln().is_decreasing(&p, &pos), Some(false));
    // SymPy: is_convex(-log(p), p, domain=Interval.open(0, oo)) is True,
    //        is_convex(1/p, p, domain=Interval.open(0, oo)) is True,
    //        is_increasing(1/p, Interval.open(0, oo), p) is False
    assert_eq!((-p.ln()).is_convex(&p, &pos), Some(true));
    assert_eq!((ctx.int(1) / &p).is_convex(&p, &pos), Some(true));
    assert_eq!((ctx.int(1) / &p).is_increasing(&p, &pos), Some(false));
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. `abs` kinks in maximum / minimum / function_range / stationary_points
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extrema_of_abs_expressions() {
    let ctx = Context::new();
    sym!(ctx; u, Real);
    let m12 = ctx.interval(&ctx.int(-1), &ctx.int(2), false, false);

    // SymPy: maximum(Abs(u), u, Interval(-1, 2)) == 2, minimum(...) == 0
    assert_eq!(s(&u.abs().maximum(&u, &m12).unwrap()), "2");
    assert_eq!(s(&u.abs().minimum(&u, &m12).unwrap()), "0");

    // SymPy: minimum(Abs(u - 1) + u**2, u, Interval(-1, 2)) == 3/4,
    //        maximum(...) == 5
    let f = (&u - 1).abs() + u.powi(2);
    assert_eq!(f.minimum(&u, &m12).unwrap(), ctx.rational(3, 4));
    assert_eq!(f.maximum(&u, &m12).unwrap(), ctx.int(5));

    // SymPy: maximum(u**2 - Abs(u), u, Interval(-2, 2)) == 2, minimum == -1/4
    let m22 = ctx.interval(&ctx.int(-2), &ctx.int(2), false, false);
    let g = u.powi(2) - u.abs();
    assert_eq!(g.maximum(&u, &m22).unwrap(), ctx.int(2));
    assert_eq!(g.minimum(&u, &m22).unwrap(), ctx.rational(-1, 4));

    // A region where the derivative vanishes identically: |u| + u = 0 on
    // [-1, 0].  SymPy: maximum(Abs(u) + u, u, Interval(-1, 2)) raises
    // NotImplementedError ("Unable to find critical points"); the values
    // are max 4 (at 2) and min 0 (on [-1, 0]).
    let h = u.abs() + &u;
    assert_eq!(h.maximum(&u, &m12).unwrap(), ctx.int(4));
    assert_eq!(h.minimum(&u, &m12).unwrap(), ctx.int(0));

    // Two kinks on ℝ.  SymPy: minimum(Abs(u) + Abs(u - 1), u, S.Reals) raises
    // NotImplementedError; |u| + |u − 1| ≥ 1 with equality on [0, 1] and
    // is unbounded above.
    let k = u.abs() + (&u - 1).abs();
    assert_eq!(k.minimum(&u, &ctx.reals()).unwrap(), ctx.int(1));
    assert_eq!(k.maximum(&u, &ctx.reals()).unwrap(), ctx.infinity());

    // A periodic kink family on a bounded domain.  SymPy:
    // maximum(Abs(sin(u)), u, Interval(0, 2*pi)) raises NotImplementedError
    // ("as_set is not implemented for relationals with periodic solutions");
    // the values are 1 and 0.
    let two_pi = ctx.interval(&ctx.int(0), &(&ctx.pi() * 2), false, false);
    assert_eq!(u.sin().abs().maximum(&u, &two_pi).unwrap(), ctx.int(1));
    assert_eq!(u.sin().abs().minimum(&u, &two_pi).unwrap(), ctx.int(0));
}

#[test]
fn function_range_of_abs_expressions() {
    let ctx = Context::new();
    sym!(ctx; x, Real);
    let m12 = ctx.interval(&ctx.int(-1), &ctx.int(2), false, false);
    let m22 = ctx.interval(&ctx.int(-2), &ctx.int(2), false, false);
    let r = |f: &Ex, d: &SetEx| s(&f.function_range(&x, d).unwrap());

    // SymPy: function_range(Abs(x), x, Interval(-1, 2)) == Interval(0, 2)
    assert_eq!(r(&x.abs(), &m12), "[0, 2]");
    // SymPy: function_range(Abs(x - 1) + x**2, x, Interval(-1, 2)) == Interval(3/4, 5)
    assert_eq!(r(&((&x - 1).abs() + x.powi(2)), &m12), "[3/4, 5]");
    // SymPy: function_range(x**2 - Abs(x), x, Interval(-2, 2)) == Interval(-1/4, 2)
    assert_eq!(r(&(x.powi(2) - x.abs()), &m22), "[-1/4, 2]");
    // SymPy: function_range(Abs(x), x, S.Reals) == Interval(0, oo)
    assert_eq!(r(&x.abs(), &ctx.reals()), "[0, oo)");
    // SymPy raises NotImplementedError for Abs(x) + x and Abs(x) + Abs(x - 1);
    // the images are [0, 4] on [-1, 2] and [1, oo) on ℝ.
    assert_eq!(r(&(x.abs() + &x), &m12), "[0, 4]");
    assert_eq!(r(&(x.abs() + (&x - 1).abs()), &ctx.reals()), "[1, oo)");
}

#[test]
fn stationary_points_resolve_sign_factors_like_sympy() {
    let ctx = Context::new();
    sym!(ctx; x, Real);
    let m12 = ctx.interval(&ctx.int(-1), &ctx.int(2), false, false);
    let sp = |f: &Ex, d: Option<&SetEx>| s(&f.stationary_points(&x, d).unwrap());

    // SymPy: stationary_points(Abs(x), x) == {0}   (sign(0) = 0)
    assert_eq!(sp(&x.abs(), None), "{0}");
    // SymPy: stationary_points(Abs(x - 1) + x**2, x, Interval(-1, 2)) == {1/2}
    //        (2x + 1 = 0 at x = -1/2 is discarded: sign(x - 1) = -1 there)
    assert_eq!(sp(&((&x - 1).abs() + x.powi(2)), Some(&m12)), "{1/2}");
    // SymPy: stationary_points(x**2 - Abs(x), x) == {-1/2, 0, 1/2}
    assert_eq!(sp(&(x.powi(2) - x.abs()), None), "{-1/2, 0, 1/2}");
    // SymPy: stationary_points(Abs(x) + x, x, Interval(-1, 2)) == Interval.Ropen(-1, 0)
    assert_eq!(sp(&(x.abs() + &x), Some(&m12)), "[-1, 0)");
}

#[test]
fn constant_in_var_is_simplified_before_reporting() {
    let ctx = Context::new();
    sym!(ctx; x, Real);
    let z1 = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    // SymPy: function_range(sin(x)**2 + cos(x)**2, x, Interval(0, 1))
    //        == {sin(x)**2 + cos(x)**2}   (left unsimplified); the image is {1}.
    let one = &x.sin().powi(2) + &x.cos().powi(2);
    assert_eq!(s(&one.function_range(&x, &z1).unwrap()), "{1}");
    assert_eq!(one.maximum(&x, &z1).unwrap(), ctx.int(1));
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Error variant: algorithmic failures are `ComputationFailed`
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn algorithmic_failures_are_computation_failed_not_not_implemented() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m1 = ctx.interval(&ctx.int(-1), &ctx.int(1), false, false);
    // A pole inside the domain.
    assert!(matches!(
        (ctx.int(1) / &x).maximum(&x, &m1),
        Err(SymplexError::ComputationFailed {
            operation: "maximum",
            ..
        })
    ));
    // An opaque node.
    assert!(matches!(
        x.floor().minimum(&x, &m1),
        Err(SymplexError::ComputationFailed {
            operation: "minimum",
            ..
        })
    ));
    // Stationary points that cannot be enumerated on an unbounded domain.
    assert!(matches!(
        x.sin().function_range(&x, &ctx.reals()),
        Err(SymplexError::ComputationFailed {
            operation: "function_range",
            ..
        })
    ));
    // Ill-formed input stays `InvalidArgument`.
    assert!(matches!(
        x.maximum(&(&x + 1), &m1),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        x.maximum(&x, &ctx.empty_set()),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. periodicity: a period, not necessarily the fundamental one
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn periodicity_reports_a_period() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    // SymPy: periodicity(sin(a*x), x) == 2*pi/Abs(a)
    let per = (&a * &x)
        .sin()
        .periodicity(&x)
        .expect("sin(a·x) is periodic");
    assert_eq!(per.equals(&(&ctx.pi() * 2 / a.abs())), Some(true), "{per}");
    // SymPy: periodicity(sin(x)**2*cos(x)**2, x) == pi/2 (SymPy simplifies
    // to sin(2x)²/4 first).  symplex returns the lcm of the component
    // periods, pi — a period, documented as not necessarily fundamental.
    let p = (&x.sin().powi(2) * &x.cos().powi(2))
        .periodicity(&x)
        .expect("periodic");
    assert_eq!(s(&p), "pi");
    // SymPy: periodicity(sin(x) + cos(2*x), x) == 2*pi
    assert_eq!(
        s(&(&x.sin() + &(&x * 2).cos()).periodicity(&x).unwrap()),
        "2*pi"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. minimal_polynomial is verified and irreducible
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn minimal_polynomial_is_verified_and_irreducible() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s2 = ctx.int(2).sqrt();
    let s3 = ctx.int(3).sqrt();
    let s5 = ctx.int(5).sqrt();
    let s8 = ctx.int(8).sqrt();
    let cbrt2 = ctx.int(2).pow(&ctx.rational(1, 3));
    let cbrt4 = ctx.int(4).pow(&ctx.rational(1, 3));

    // SymPy 1.14 minimal_polynomial(…, x):
    let cases: Vec<(&str, Ex, Ex)> = vec![
        // sqrt(2)*sqrt(2)          -> x - 2
        ("√2·√2", &s2 * &s2, &x - 2),
        // sqrt(2) + sqrt(8)        -> x**2 - 18
        ("√2 + √8", &s2 + &s8, &x.powi(2) - 18),
        // sqrt(2) + sqrt(3)        -> x**4 - 10*x**2 + 1
        ("√2 + √3", &s2 + &s3, &x.powi(4) - &x.powi(2) * 10 + 1),
        // cbrt(2)                  -> x**3 - 2
        ("∛2", cbrt2.clone(), &x.powi(3) - 2),
        // 1/(1 + sqrt(2))          -> x**2 + 2*x - 1
        (
            "(1+√2)⁻¹",
            (ctx.int(1) + &s2).powi(-1),
            &x.powi(2) + &x * 2 - 1,
        ),
        // GoldenRatio              -> x**2 - x - 1
        ("φ", ctx.golden_ratio(), &x.powi(2) - &x - 1),
        // (1 + sqrt(5))/2          -> x**2 - x - 1
        ("(1+√5)/2", (ctx.int(1) + &s5) / 2, &x.powi(2) - &x - 1),
        // sqrt(2)*sqrt(3)          -> x**2 - 6
        ("√2·√3", &s2 * &s3, &x.powi(2) - 6),
        // cbrt(2) + cbrt(4)        -> x**3 - 6*x - 6
        ("∛2 + ∛4", &cbrt2 + &cbrt4, &x.powi(3) - &x * 6 - 6),
        // sqrt(2) + sqrt(3) + sqrt(5) -> x**8 - 40*x**6 + 352*x**4 - 960*x**2 + 576
        (
            "√2 + √3 + √5",
            &s2 + &s3 + &s5,
            &x.powi(8) - &x.powi(6) * 40 + &x.powi(4) * 352 - &x.powi(2) * 960 + 576,
        ),
        // I + sqrt(2)              -> x**4 - 2*x**2 + 9
        (
            "i + √2",
            ctx.i_unit() + &s2,
            &x.powi(4) - &x.powi(2) * 2 + 9,
        ),
        // sqrt(2) - sqrt(2)        -> x
        ("√2 − √2", &s2 - &s2, x.clone()),
    ];
    for (label, alpha, expected) in cases {
        let m = alpha
            .minimal_polynomial(&x)
            .unwrap_or_else(|| panic!("{label}: minimal polynomial not found"));
        assert_eq!(m, expected, "{label}: got {m}");
        assert_eq!(
            m.is_irreducible(&x),
            Some(true),
            "{label}: {m} is reducible"
        );
        // The polynomial vanishes at the number (16 digits).
        if alpha.is_real() != Some(false)
            && let Ok(v) = m.subs(&x, &alpha).eval_f64()
        {
            assert!(v.abs() < 1e-9, "{label}: m(α) = {v}");
        }
    }
}

#[test]
fn minimal_polynomial_none_when_unverifiable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // A fractional power of a negative rational: the branch is not decided
    // here, so nothing is guessed.  (SymPy 1.14: minimal_polynomial
    // ((-8)**Rational(1, 3), x) == x**3 + 8 with the default compose=True —
    // reducible, (x + 2)(x**2 - 2*x + 4) — and x**2 - 2*x + 4 with
    // compose=False, the polynomial of its principal value 1 + √3·i.)
    assert!(
        ctx.int(-8)
            .pow(&ctx.rational(1, 3))
            .minimal_polynomial(&x)
            .is_none()
    );
    // A square root of a negative algebraic number (√2 − 2 < 0): likewise.
    // (SymPy: minimal_polynomial(sqrt(sqrt(2) - 2), x) == x**4 + 4*x**2 + 2.)
    assert!(
        (ctx.int(2).sqrt() - 2)
            .sqrt()
            .minimal_polynomial(&x)
            .is_none()
    );
    // Transcendental input is still `None`.
    assert!(ctx.pi().minimal_polynomial(&x).is_none());
    assert!((ctx.e() + 1).minimal_polynomial(&x).is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. RootOf indices agree with the evaluator and are verified
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn real_roots_of_sextic_with_interleaved_complex_pair() {
    // x**6 - 3*x**2 + 1 is irreducible with four real roots and a complex
    // pair ±1.3709…i whose real part 0 lies between them, so the (re, im)
    // order of the six roots is r0 < r1 < (0, -1.37i) < (0, +1.37i) < r2 < r3.
    // SymPy: real_roots(x**6 - 3*x**2 + 1) == [CRootOf(f, 0), CRootOf(f, 1),
    //        CRootOf(f, 2), CRootOf(f, 3)] (SymPy indexes real roots first);
    //        their 30-digit values are
    //        -1.23777578189184008222625199711, -0.589318551662732485630016172721,
    //         0.589318551662732485630016172721,  1.23777578189184008222625199711.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(6) - &x.powi(2) * 3 + 1;
    assert_eq!(f.is_irreducible(&x), Some(true));
    let roots = f
        .real_roots(&x)
        .expect("polynomial with rational coefficients");
    assert_eq!(
        roots.iter().map(s).collect::<Vec<_>>(),
        [
            "RootOf(x^6 - 3*x^2 + 1, 0)",
            "RootOf(x^6 - 3*x^2 + 1, 1)",
            "RootOf(x^6 - 3*x^2 + 1, 4)",
            "RootOf(x^6 - 3*x^2 + 1, 5)",
        ]
    );
    let expected = [
        -1.237_775_781_891_84,
        -0.589_318_551_662_732_5,
        0.589_318_551_662_732_5,
        1.237_775_781_891_84,
    ];
    for (r, e) in roots.iter().zip(expected) {
        let v = r.eval_f64().unwrap();
        assert!((v - e).abs() < 1e-12, "{r} = {v}, expected {e}");
        assert!(f.subs(&x, r).eval_f64().unwrap().abs() < 1e-9);
    }
    // The same index is used at higher precision (30 digits): the order is
    // not a rounding artefact.
    let hi = roots[2].eval_decimal(30).unwrap();
    assert!(hi.starts_with("0.58931855166273248563"), "{hi}");
}

#[test]
fn real_roots_indices_match_the_evaluator_for_other_shapes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // SymPy: real_roots(x**3 - 3*x**2 + 4*x - 1) == [CRootOf(…, 0)]
    //        ≈ 0.31767219617198067263 (one real root; the complex pair
    //        1.34116 ± 1.16154i lies to its right → index 0).
    let g = &x.powi(3) - &x.powi(2) * 3 + &x * 4 - 1;
    let roots = g.real_roots(&x).unwrap();
    assert_eq!(roots.len(), 1);
    assert_eq!(s(&roots[0]), "RootOf(x^3 - 3*x^2 + 4*x - 1, 0)");
    assert!((roots[0].eval_f64().unwrap() - 0.317_672_196_171_980_67).abs() < 1e-12);
    // SymPy: real_roots(x**3 - x - 1) == [CRootOf(x**3 - x - 1, 0)]
    //        ≈ 1.3247179572447460260 (the complex pair -0.662359 ± 0.562279i
    //        lies to its left → index 2 in (re, im) order).
    let h = &x.powi(3) - &x - 1;
    let roots = h.real_roots(&x).unwrap();
    assert_eq!(s(&roots[0]), "RootOf(x^3 - x - 1, 2)");
    assert!((roots[0].eval_f64().unwrap() - 1.324_717_957_244_746).abs() < 1e-12);
    // Rational roots stay exact and a reducible input names each factor.
    // SymPy: real_roots((x - 1)*(x**2 - 2*x + 2)) == [1]
    let k = (&x - 1) * (&x.powi(2) - &x * 2 + 2);
    assert_eq!(k.real_roots(&x).unwrap(), vec![ctx.int(1)]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Long computations no longer run under the arena write lock
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn algebraic_methods_run_concurrently_on_a_shared_context() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let (ctx, x, y) = (ctx.clone(), x.clone(), y.clone());
            thread::spawn(move || {
                for _ in 0..3 {
                    let a = ctx.int(2 + i).sqrt() + ctx.int(3).sqrt();
                    let m = a.minimal_polynomial(&x).expect("algebraic");
                    assert_eq!(m.is_irreducible(&x), Some(true));
                    let f = &x.powi(5) - &x - (1 + i);
                    assert_eq!(f.real_roots(&x).unwrap().len(), 1);
                    let g = (&x.powi(2) - &y.powi(2)).gcd_all(&(&x - &y)).unwrap();
                    assert_eq!(g, &x - &y);
                    let basis = Ex::groebner(
                        &[&x.powi(2) + &y.powi(2) - 1, &x - &y],
                        &[x.clone(), y.clone()],
                        symplex::multipoly::MonomialOrder::Lex,
                    )
                    .unwrap();
                    assert_eq!(basis.len(), 2);
                    let r = x
                        .powi(2)
                        .reduce_modulo(
                            &basis,
                            &[x.clone(), y.clone()],
                            symplex::multipoly::MonomialOrder::Lex,
                        )
                        .unwrap();
                    assert_eq!(r, ctx.rational(1, 2));
                    let (_, fs) = (&x.powi(2) + 1).factor_mod(&x, 5).unwrap();
                    assert_eq!(fs.len(), 2);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("thread panicked");
    }
}
