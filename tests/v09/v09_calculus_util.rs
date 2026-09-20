//! symplex 0.9 — function analysis on `Ex` (SymPy `calculus.util` /
//! `calculus.singularities`): `singularities`, `stationary_points`,
//! `maximum` / `minimum`, `is_increasing` & friends, `periodicity`,
//! `is_convex`, `function_range`.
//!
//! Reference values are from SymPy 1.14 (`symplex/.venv`), quoted in the
//! comments of each test.  Where symplex deliberately differs from SymPy
//! (strict monotonicity of `x³`, `is_monotonic`, the fundamental period of
//! `sin(x)²`) the SymPy answer is quoted too.

use symplex::prelude::*;

fn s<T: std::fmt::Display>(e: &T) -> String {
    format!("{e}")
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. singularities
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn singularities_rational_log_tan_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // SymPy: singularities(1/(x**2 - 1), x) == {-1, 1}
    let f = 1 / (&x.powi(2) - 1);
    assert_eq!(s(&f.singularities(&x, None).unwrap()), "{-1, 1}");

    // SymPy: singularities(log(x), x) == {0}
    let l = x.ln().singularities(&x, None).unwrap();
    assert_eq!(s(&l), "{0}");
    assert_eq!(l.contains(&ctx.zero()), Some(true));

    // SymPy: singularities(1/x, x) == {0}
    assert_eq!(s(&(1 / &x).singularities(&x, None).unwrap()), "{0}");

    // SymPy: singularities(x**2, x) == EmptySet
    assert_eq!(
        x.powi(2).singularities(&x, None).unwrap().is_empty(),
        Some(true)
    );

    // Restricted to a domain: {-1, 1} ∩ [0, 5] = {1}
    let dom = ctx.interval(&ctx.int(0), &ctx.int(5), false, false);
    assert_eq!(s(&f.singularities(&x, Some(&dom)).unwrap()), "{1}");

    // SymPy: singularities(tan(x), x) is the union of the two ImageSets
    // pi/2 + 2*n*pi and 3*pi/2 + 2*n*pi.  Enumerated on a bounded domain:
    // [0, 10] contains pi/2, 3*pi/2, 5*pi/2.
    let ten = ctx.interval(&ctx.int(0), &ctx.int(10), false, false);
    let poles = x.tan().singularities(&x, Some(&ten)).unwrap();
    let pts = poles.as_finite_set().expect("finite on a bounded domain");
    assert_eq!(pts.len(), 3, "{poles}");
    let pi = ctx.pi();
    for (k, p) in pts.iter().enumerate() {
        let expected = &pi * (2 * k as i64 + 1) / 2;
        assert_eq!(p.equals(&expected), Some(true), "{p} vs {expected}");
    }
    // On ℝ the family is kept as a condition set, never silently truncated.
    let all = x.tan().singularities(&x, None).unwrap();
    assert!(all.as_finite_set().is_none(), "{all}");
    assert_eq!(all.contains(&(&pi / 2)), Some(true), "{all}");
    assert_eq!(all.contains(&ctx.int(1)), Some(false), "{all}");

    // Only real points: 1/(x² + 1) has none on ℝ.
    assert_eq!(
        (1 / (&x.powi(2) + 1))
            .singularities(&x, None)
            .unwrap()
            .is_empty(),
        Some(true)
    );
    // `var` must be a symbol.
    assert!(matches!(
        f.singularities(&(&x + 1), None),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. stationary_points
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn stationary_points_cubic_and_sine() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) - &x * 3;

    // SymPy: stationary_points(x**3 - 3*x, x) == {-1, 1}
    assert_eq!(s(&f.stationary_points(&x, None).unwrap()), "{-1, 1}");
    // SymPy: stationary_points(x**3 - 3*x, x, Interval(0, 5)) == {1}
    let dom = ctx.interval(&ctx.int(0), &ctx.int(5), false, false);
    assert_eq!(s(&f.stationary_points(&x, Some(&dom)).unwrap()), "{1}");

    // SymPy: stationary_points(sin(x), x, Interval(0, 2*pi)) == {pi/2, 3*pi/2}
    let pi = ctx.pi();
    let two_pi = ctx.interval(&ctx.int(0), &(&pi * 2), false, false);
    let sp = x.sin().stationary_points(&x, Some(&two_pi)).unwrap();
    let pts = sp.as_finite_set().unwrap();
    assert_eq!(pts.len(), 2, "{sp}");
    assert_eq!(pts[0].equals(&(&pi / 2)), Some(true), "{}", pts[0]);
    assert_eq!(pts[1].equals(&(&pi * 3 / 2)), Some(true), "{}", pts[1]);

    // exp(x) has no stationary points; a constant is stationary everywhere.
    assert_eq!(
        x.exp().stationary_points(&x, None).unwrap().is_empty(),
        Some(true)
    );
    assert_eq!(ctx.int(7).stationary_points(&x, Some(&dom)).unwrap(), dom);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. maximum / minimum
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn maximum_minimum_on_intervals() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) - &x * 3;
    let dom = ctx.interval(&ctx.int(-2), &ctx.int(2), false, false);

    // SymPy: maximum(x**3 - 3*x, x, Interval(-2, 2)) == 2, minimum == -2
    assert_eq!(s(&f.maximum(&x, &dom).unwrap()), "2");
    assert_eq!(s(&f.minimum(&x, &dom).unwrap()), "-2");

    // SymPy: maximum(x**2, x, S.Reals) == oo, minimum == 0
    assert_eq!(x.powi(2).maximum(&x, &ctx.reals()).unwrap(), ctx.infinity());
    assert_eq!(s(&x.powi(2).minimum(&x, &ctx.reals()).unwrap()), "0");

    // SymPy: minimum(1/x, x, Interval(1, oo)) == 0 (limit), maximum == 1
    let tail = ctx.interval(&ctx.int(1), &ctx.infinity(), false, true);
    assert_eq!(s(&(1 / &x).minimum(&x, &tail).unwrap()), "0");
    assert_eq!(s(&(1 / &x).maximum(&x, &tail).unwrap()), "1");

    // SymPy: maximum(sin(x), x, Interval(0, pi)) == 1
    let zero_pi = ctx.interval(&ctx.int(0), &ctx.pi(), false, false);
    assert_eq!(s(&x.sin().maximum(&x, &zero_pi).unwrap()), "1");
    assert_eq!(s(&x.sin().minimum(&x, &zero_pi).unwrap()), "0");

    // SymPy: maximum(x, x, Interval.open(0, 1)) == 1 (supremum, not attained)
    let unit = ctx.interval(&ctx.int(0), &ctx.int(1), true, true);
    assert_eq!(s(&x.maximum(&x, &unit).unwrap()), "1");
    assert_eq!(s(&x.minimum(&x, &unit).unwrap()), "0");

    // Union of intervals: x² on [-1, 1] ∪ [2, 3] → max 9, min 0.
    let two = ctx.interval(&ctx.int(-1), &ctx.int(1), false, false);
    let three = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    let both = two.union(&three);
    assert_eq!(s(&x.powi(2).maximum(&x, &both).unwrap()), "9");
    assert_eq!(s(&x.powi(2).minimum(&x, &both).unwrap()), "0");

    // A constant.
    assert_eq!(s(&ctx.int(5).maximum(&x, &dom).unwrap()), "5");

    // Errors: singularity inside the domain, empty / non-interval domain,
    // discontinuous function, limit that does not exist.  (0.11.1: the
    // algorithmic failures are `ComputationFailed`, not `NotImplemented`.)
    let across_zero = ctx.interval(&ctx.int(-1), &ctx.int(1), false, false);
    assert!(matches!(
        (1 / &x).maximum(&x, &across_zero),
        Err(SymplexError::ComputationFailed { .. })
    ));
    assert!(matches!(
        x.maximum(&x, &ctx.empty_set()),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let cond = x.sin().solve_ge(&x); // ConditionSet: not a union of intervals
    assert!(matches!(
        x.maximum(&x, &cond),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        x.floor().maximum(&x, &dom),
        Err(SymplexError::ComputationFailed { .. })
    ));
    assert!(matches!(
        x.sin().maximum(&x, &ctx.reals()),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. monotonicity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn monotonicity_queries() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let reals = ctx.reals();
    let half = ctx.interval(&ctx.int(0), &ctx.infinity(), false, true); // [0, oo)
    let open_half = ctx.interval(&ctx.int(0), &ctx.infinity(), true, true); // (0, oo)
    let left = ctx.interval(&ctx.neg_infinity(), &ctx.int(0), true, false); // (-oo, 0]

    // SymPy: is_increasing(x**3, S.Reals, x) is True
    assert_eq!(x.powi(3).is_increasing(&x, &reals), Some(true));
    // SymPy: is_strictly_increasing(x**3, S.Reals, x) is None (it tests
    // R ⊆ {3x² > 0}); x³ is strictly increasing, so symplex says Some(true).
    assert_eq!(x.powi(3).is_strictly_increasing(&x, &reals), Some(true));
    // SymPy: is_increasing(x**2, S.Reals, x) is False
    assert_eq!(x.powi(2).is_increasing(&x, &reals), Some(false));
    // SymPy: is_increasing(x**2, Interval(0, oo), x) is True
    assert_eq!(x.powi(2).is_increasing(&x, &half), Some(true));
    // SymPy: is_strictly_increasing(x**2, Interval.open(0, oo), x) is True
    assert_eq!(x.powi(2).is_strictly_increasing(&x, &open_half), Some(true));
    // SymPy gives False for the closed half-line (2x = 0 at 0); x² is
    // nevertheless strictly increasing on [0, oo).
    assert_eq!(x.powi(2).is_strictly_increasing(&x, &half), Some(true));
    // SymPy: is_decreasing(x**2, Interval(-oo, 0), x) is True
    assert_eq!(x.powi(2).is_decreasing(&x, &left), Some(true));
    assert_eq!(x.powi(2).is_strictly_decreasing(&x, &left), Some(true));
    assert_eq!(x.powi(2).is_decreasing(&x, &reals), Some(false));

    // SymPy: is_increasing(1/x, Interval.open(0, oo), x) is False,
    //        is_decreasing(1/x, Interval.open(0, oo), x) is True
    let inv = 1 / &x;
    assert_eq!(inv.is_increasing(&x, &open_half), Some(false));
    assert_eq!(inv.is_decreasing(&x, &open_half), Some(true));
    assert_eq!(inv.is_strictly_decreasing(&x, &open_half), Some(true));
    // 1/x is not monotonic across its pole.
    assert_eq!(inv.is_decreasing(&x, &reals), Some(false));

    // SymPy: is_increasing(exp(x), S.Reals, x) is True
    assert_eq!(x.exp().is_increasing(&x, &reals), Some(true));
    // SymPy: is_increasing(-x, S.Reals, x) is False
    assert_eq!((-&x).is_increasing(&x, &reals), Some(false));
    assert_eq!((-&x).is_decreasing(&x, &reals), Some(true));

    // is_monotonic = increasing ∨ decreasing (three-valued).
    // SymPy's is_monotonic asks for a derivative without zeros and says
    // False for x**3; symplex reports the monotonicity itself.
    assert_eq!(x.powi(3).is_monotonic(&x, &reals), Some(true));
    // SymPy: is_monotonic(x**2, S.Reals, x) is False
    assert_eq!(x.powi(2).is_monotonic(&x, &reals), Some(false));
    // x·exp(x): the sign of (x + 1)·exp(x) on ℝ is not decided by the
    // inequality solver — undecided, not guessed.  (SymPy: False.)
    let xex = &x * &x.exp();
    let m = xex.is_monotonic(&x, &reals);
    assert!(m.is_none() || m == Some(false), "{m:?}");
    assert_ne!(xex.is_increasing(&x, &reals), Some(true));

    // Constants: increasing and decreasing, but not strictly.
    assert_eq!(ctx.int(3).is_increasing(&x, &reals), Some(true));
    assert_eq!(ctx.int(3).is_strictly_increasing(&x, &reals), Some(false));
    // Empty domain: vacuous.
    assert_eq!(x.powi(2).is_increasing(&x, &ctx.empty_set()), Some(true));

    // A derivative undefined at a closed endpoint is tested on the interior
    // once the function is known to be continuous there: sqrt(x) on [0, oo).
    // (SymPy's derivative-only criterion answers False here.)
    assert_eq!(x.sqrt().is_increasing(&x, &half), Some(true));
    assert_eq!(x.sqrt().is_strictly_increasing(&x, &half), Some(true));
    assert_eq!(x.sqrt().is_decreasing(&x, &half), Some(false));
    // Discontinuous nodes are never decided from the formal derivative.
    assert_eq!(x.floor().is_increasing(&x, &reals), None);
    assert_eq!(x.floor().is_decreasing(&x, &reals), None);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. periodicity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn periodicity_of_trigonometric_expressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = |e: &Ex| e.periodicity(&x).map(|p| s(&p));

    // SymPy: periodicity(sin(2*x) + cos(3*x), x) == 2*pi
    assert_eq!(p(&(&(&x * 2).sin() + &(&x * 3).cos())), Some("2*pi".into()));
    // SymPy: periodicity(tan(x), x) == pi
    assert_eq!(p(&x.tan()), Some("pi".into()));
    // SymPy: periodicity(x**2, x) is None
    assert_eq!(p(&x.powi(2)), None);
    // SymPy: periodicity(S(3), x) == 0
    assert_eq!(p(&ctx.int(3)), Some("0".into()));
    // SymPy: periodicity(sin(x), x) == 2*pi
    assert_eq!(p(&x.sin()), Some("2*pi".into()));
    // SymPy: periodicity(sin(3*x + 1), x) == 2*pi/3
    assert_eq!(p(&(&x * 3 + 1).sin()), Some("2/3*pi".into()));
    // SymPy: periodicity(sin(x/2), x) == 4*pi
    assert_eq!(p(&(&x / 2).sin()), Some("4*pi".into()));
    // SymPy: periodicity(sin(x) + cos(x/3), x) == 6*pi
    assert_eq!(p(&(&x.sin() + &(&x / 3).cos())), Some("6*pi".into()));
    // SymPy: periodicity(sin(pi*x), x) == 2
    assert_eq!(p(&(&x * &ctx.pi()).sin()), Some("2".into()));
    // SymPy: periodicity(2*sin(x), x) == 2*pi; periodicity(sin(x) + 1, x) == 2*pi
    assert_eq!(p(&(&x.sin() * 2)), Some("2*pi".into()));
    assert_eq!(p(&(&x.sin() + 1)), Some("2*pi".into()));
    // SymPy: periodicity(exp(sin(x)), x) == 2*pi   (composition)
    assert_eq!(p(&x.sin().exp()), Some("2*pi".into()));
    // SymPy: periodicity(sin(x)*cos(x), x) == pi; periodicity(cot(x), x) == pi;
    //        periodicity(cos(x)/sin(x), x) == pi
    assert_eq!(p(&(&x.sin() * &x.cos())), Some("pi".into()));
    assert_eq!(p(&x.cot()), Some("pi".into()));
    assert_eq!(p(&(&x.cos() / &x.sin())), Some("pi".into()));
    // SymPy: periodicity(sec(x), x) == 2*pi
    assert_eq!(p(&x.sec()), Some("2*pi".into()));
    // SymPy: periodicity(sin(x)*sin(2*x), x) == 2*pi
    assert_eq!(p(&(&x.sin() * &(&x * 2).sin())), Some("2*pi".into()));
    // SymPy: periodicity(sin(x)**2, x) == 2*pi — a period, but the
    // fundamental period of sin² is pi, which is what symplex returns.
    assert_eq!(p(&x.sin().powi(2)), Some("pi".into()));
    // SymPy: periodicity(sin(x)**2 + cos(x)**2, x) == 0   (simplifies to 1)
    assert_eq!(p(&(&x.sin().powi(2) + &x.cos().powi(2))), Some("0".into()));
    // SymPy: periodicity(sin(x) + x, x) is None; periodicity(sin(x**2), x) is
    // None; periodicity(sin(x)*exp(x), x) is None;
    // periodicity(sin(sqrt(2)*x) + sin(x), x) is None
    assert_eq!(p(&(&x.sin() + &x)), None);
    assert_eq!(p(&x.powi(2).sin()), None);
    assert_eq!(p(&(&x.sin() * &x.exp())), None);
    assert_eq!(p(&(&(&x * &ctx.int(2).sqrt()).sin() + &x.sin())), None);
    // Another variable is a constant.
    let y = ctx.symbol("y");
    assert_eq!(p(&y.sin()), Some("0".into()));
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. is_convex
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convexity_queries() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let reals = ctx.reals();

    // SymPy: is_convex(x**2, x) is True; is_convex(x**3, x) is False
    assert_eq!(x.powi(2).is_convex(&x, &reals), Some(true));
    assert_eq!(x.powi(3).is_convex(&x, &reals), Some(false));
    // SymPy: is_convex(x**3, x, domain=Interval(0, oo)) is True
    let half = ctx.interval(&ctx.int(0), &ctx.infinity(), false, true);
    assert_eq!(x.powi(3).is_convex(&x, &half), Some(true));
    // SymPy: is_convex(exp(x), x) is True; is_convex(-x**2, x) is False
    assert_eq!(x.exp().is_convex(&x, &reals), Some(true));
    assert_eq!((-&x.powi(2)).is_convex(&x, &reals), Some(false));
    // Affine functions are convex; 1/x is convex on (0, oo).
    assert_eq!((&x * 2 + 1).is_convex(&x, &reals), Some(true));
    let pos = ctx.interval(&ctx.int(0), &ctx.infinity(), true, true);
    assert_eq!((1 / &x).is_convex(&x, &pos), Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. function_range
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn function_range_tracks_open_and_closed_endpoints() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let r = |f: &Ex, d: &SetEx| s(&f.function_range(&x, d).unwrap());

    // SymPy: function_range(sin(x), x, Interval(0, pi)) == Interval(0, 1)
    assert_eq!(
        r(&x.sin(), &ctx.interval(&ctx.int(0), &pi, false, false)),
        "[0, 1]"
    );
    // SymPy: function_range(x**2, x, Interval(-1, 2)) == Interval(0, 4)
    assert_eq!(
        r(
            &x.powi(2),
            &ctx.interval(&ctx.int(-1), &ctx.int(2), false, false)
        ),
        "[0, 4]"
    );
    // SymPy: function_range(x**2, x, S.Reals) == Interval(0, oo)
    assert_eq!(r(&x.powi(2), &ctx.reals()), "[0, oo)");
    // SymPy: function_range(1/x, x, Interval(1, oo)) == Interval.Lopen(0, 1)
    let tail = ctx.interval(&ctx.int(1), &ctx.infinity(), false, true);
    assert_eq!(r(&(1 / &x), &tail), "(0, 1]");
    // SymPy: function_range(x, x, Interval.open(0, 1)) == Interval.open(0, 1)
    assert_eq!(
        r(&x, &ctx.interval(&ctx.int(0), &ctx.int(1), true, true)),
        "(0, 1)"
    );
    // SymPy: function_range(x**2, x, Interval.open(-1, 1)) == Interval.Ropen(0, 1)
    assert_eq!(
        r(
            &x.powi(2),
            &ctx.interval(&ctx.int(-1), &ctx.int(1), true, true)
        ),
        "[0, 1)"
    );
    // SymPy: function_range(x**3 - 3*x, x, Interval(-2, 2)) == Interval(-2, 2)
    let f = &x.powi(3) - &x * 3;
    assert_eq!(
        r(&f, &ctx.interval(&ctx.int(-2), &ctx.int(2), false, false)),
        "[-2, 2]"
    );
    // SymPy: function_range(exp(x), x, S.Reals) == Interval.open(0, oo)
    assert_eq!(r(&x.exp(), &ctx.reals()), "(0, oo)");
    // SymPy: function_range(tan(x), x, Interval.open(-pi/2, pi/2)) == Interval(-oo, oo)
    let branch = ctx.interval(&(-&pi / 2), &(&pi / 2), true, true);
    assert_eq!(r(&x.tan(), &branch), "(-oo, oo)");
    // SymPy: function_range(x**2, x, Union(Interval(-1, 1), Interval(2, 3)))
    //        == Union(Interval(0, 1), Interval(4, 9))
    let two = ctx.interval(&ctx.int(-1), &ctx.int(1), false, false);
    let three = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    assert_eq!(r(&x.powi(2), &two.union(&three)), "[0, 1] ∪ [4, 9]");
    // A constant maps to a single point; the empty domain to the empty set.
    assert_eq!(r(&ctx.int(3), &ctx.reals()), "{3}");
    assert_eq!(
        ctx.int(3)
            .function_range(&x, &ctx.empty_set())
            .unwrap()
            .is_empty(),
        Some(true)
    );
}
