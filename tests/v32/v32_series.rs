//! Series with a logarithmic singularity: `x^x`, `x·ln x`, `ln x` inside
//! other functions.

use std::time::{Duration, Instant};
use symplex::prelude::*;

/// The nightly `fuzz_calculus` slow unit of 2026-09-30: `series(sinh(x^x),
/// x, 0, 3)` has no two-sided expansion (`x^x = e^(x·ln x)` is
/// `1 + x·ln x + …` from the right and complex from the left), and it took
/// 0.4 s to say so (10 s on CI's sanitizer build; the same class turned the
/// nightly red on 2026-09-28).  The logarithmic singularity fell to the
/// differentiation fallback, which asked the limit engine for the limits of
/// ever larger derivatives, again for each of five pole-retry multipliers;
/// `atan(x^(x^2))` took 2.8 s.  Now the two log-extended one-sided
/// expansions decide at once.
///
/// SymPy: `series(sinh(x**x), x, 0, 3)` → `PoleError` (its default
/// direction is `+`, where SymPy has no rule for `sinh` of a log-extended
/// argument).
#[test]
fn logarithmic_singularities_refuse_at_once() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let outer = [
        "sinh(G)",
        "sin(G)",
        "exp(G)",
        "asin(G)",
        "atan(G)",
        "ln(1 + G)",
    ];
    let inner = [
        "x^x",
        "x*ln(x)",
        "abs(x)^(1/3)",
        "sqrt(x)",
        "exp(-1/x)",
        "x^x - 1",
        "x^(x^2)",
        "ln(x)",
    ];
    for f in outer {
        for g in inner {
            let src = f.replace('G', &format!("({g})"));
            let e = ctx.parse(&src).unwrap();
            for dir in [Direction::Both, Direction::Right, Direction::Left] {
                let t = Instant::now();
                let s = e.series_dir(&x, &zero, 3, dir);
                assert!(
                    t.elapsed() < Duration::from_secs(2),
                    "{src} {dir:?} took {:?}",
                    t.elapsed()
                );
                // (sinh(ln x) = (x − 1/x)/2 and exp(ln x) = x do expand.)
                if dir == Direction::Both && matches!(g, "x^x" | "x*ln(x)" | "x^x - 1" | "x^(x^2)")
                {
                    assert!(s.has_unevaluated(), "{src}: {s}");
                }
            }
        }
    }
    let t = Instant::now();
    assert!(
        ctx.parse("sinh(x^x)")
            .unwrap()
            .try_series(&x, &zero, 3)
            .is_err()
    );
    assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
}

/// Logarithms that cancel still give a two-sided expansion: it is the
/// common expansion of the two log-extended one-sided ones, when neither
/// has a logarithm up to the requested order.  Before, these went through
/// the differentiation fallback (the same answers, more slowly).
///
/// SymPy: `series(log(sin(x)) - log(x), x, 0, 4)` → `-x**2/6 + O(x**4)`
/// (both directions); `series(log(tan(x)) - log(x), x, 0, 4)` →
/// `x**2/3 + O(x**4)`; `series(x*sinh(log(x)), x, 0, 3)` →
/// `-1/2 + x**2/2 + O(x**3)`.
#[test]
fn cancelling_logarithms_expand_two_sided() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    for (src, n, want) in [
        ("ln(sin(x)) - ln(x)", 4, "-1/6*x^2"),
        ("ln(tan(x)) - ln(x)", 4, "1/3*x^2"),
        ("cos(abs(x))*(ln(sin(x)) - ln(x))", 4, "-1/6*x^2"),
        ("ln(exp(x) - 1) - ln(x)", 3, "1/24*x^2 + 1/2*x"),
        ("x*sinh(ln(x))", 3, "1/2*x^2 - 1/2"),
        ("sinh(ln(x))", 3, "-1/(2*x) + 1/2*x"),
        ("x^5*ln(x)", 3, "0"),
    ] {
        let e = ctx.parse(src).unwrap();
        for dir in [Direction::Both, Direction::Right, Direction::Left] {
            let s = e.series_dir(&x, &zero, n, dir);
            assert_eq!(s.to_string(), want, "{src} {dir:?}");
        }
    }
    // x³·ln|x| is not O(x³): no two-sided expansion to order 3, while each
    // one-sided (log-extended) one is 0 + O(x³·ln x).
    let e = ctx.parse("x^3*ln(x)").unwrap();
    assert!(e.try_series(&x, &zero, 3).is_err());
    assert_eq!(
        e.series_dir(&x, &zero, 3, Direction::Right).to_string(),
        "0"
    );
    // ln(x²) − 2·ln(x) is 0 from the right and −2πi from the left.
    let e = ctx.parse("ln(x^2) - 2*ln(x)").unwrap();
    assert!(e.try_series(&x, &zero, 3).is_err());
    assert_eq!(
        e.series_dir(&x, &zero, 3, Direction::Left).to_string(),
        "-2*pi*I"
    );
}

/// Expansions from the left are log-extended in `ln(−x)`, as SymPy's
/// `dir='-'` (they went through a Puiseux substitution before and came out
/// as `−x·(−ln(−x) − πi)`).
///
/// SymPy: `series(x**x, x, 0, 3, dir='-')` → `1 - x*(-log(-x) - I*pi) +
/// x**2*(log(-x)**2/2 + I*pi*log(-x) - pi**2/2) + O(…)`.
#[test]
fn series_from_the_left_is_log_extended() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let s = ctx
        .parse("x^x")
        .unwrap()
        .series_dir(&x, &zero, 3, Direction::Left);
    assert_eq!(
        s.to_string(),
        "1/2*x^2*(pi*I + ln(-x))^2 + x*(pi*I + ln(-x)) + 1"
    );
    // At −10⁻⁴ the residual is O(x³·ln³|x|).
    let xv = ctx.rational(-1, 10_000);
    let resid = (&ctx.parse("x^x").unwrap() - &s)
        .subs(&x, &xv)
        .eval_complex64()
        .unwrap();
    assert!(resid.norm() < 1e-9, "{resid}");
    // From the right, unchanged: SymPy `series(x**x, x, 0, 3)` →
    // `1 + x*log(x) + x**2*log(x)**2/2 + O(…)`.
    let s = ctx
        .parse("x^x")
        .unwrap()
        .series_dir(&x, &zero, 3, Direction::Right);
    assert_eq!(s.to_string(), "1/2*x^2*ln(x)^2 + x*ln(x) + 1");
}

/// A function without a series rule of an argument with a logarithm is
/// expanded by its Taylor coefficients at the constant term (as SymPy's
/// `Function._eval_nseries` expands around a logarithmic argument); it was
/// refused after up to 2.8 s.
///
/// mpmath (dps 40): at `x = 10⁻³` the residual of the order-5 expansion
/// below is `3.3·10²·x⁵`, at `10⁻⁵` `4.2·10³·x⁵` (the `ln⁵ x` growth of an
/// `O(x⁵·ln⁵ x)` remainder).  SymPy refuses (`PoleError`).
#[test]
fn function_of_a_logarithmic_argument_from_one_side() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let e = ctx.parse("atan(x^x)").unwrap();
    let s = e.series_dir(&x, &zero, 5, Direction::Right);
    assert_eq!(s.to_string(), "-1/12*x^3*ln(x)^3 + 1/2*x*ln(x) + 1/4*pi");
    for xv in [ctx.rational(1, 1000), ctx.rational(1, 100_000)] {
        let resid = (&e - &s).subs(&x, &xv).eval_f64().unwrap();
        let x5 = xv.eval_f64().unwrap().powi(5);
        assert!(resid.abs() < 1e4 * x5, "{xv}: {resid}");
    }
    // Still refused where f is singular at the constant term, or the
    // argument itself tends to infinity: tan(π/2 + x·ln x), atan(ln x).
    for src in ["tan(pi/2 + x*ln(x))", "atan(ln(x))", "acos(ln(x))"] {
        let t = Instant::now();
        let r = ctx
            .parse(src)
            .unwrap()
            .series_dir(&x, &zero, 3, Direction::Right);
        assert!(r.has_unevaluated(), "{src}: {r}");
        assert!(
            t.elapsed() < Duration::from_secs(2),
            "{src}: {:?}",
            t.elapsed()
        );
    }
}
