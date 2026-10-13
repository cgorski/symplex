//! Equation solving against SymPy's solver test suite (0.47, agent C):
//! the cases of `sympy/solvers/tests/test_solvers.py`, `test_solveset.py`,
//! `test_inequalities.py` and `sympy/polys/tests/test_polyroots.py` that
//! symplex answered wrongly or refused, and the families around them.
//! Each test says what was wrong (or refused) before and cites its
//! oracle: the solutions substituted back, a real-root scan (sign changes
//! of the expression), SymPy 1.14 (`solve`, `solveset`, `roots`) or mpmath
//! 1.3 — SymPy as a second opinion only, never the ground truth (several
//! of its answers here are wrong, and the tests say so).

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn real_x(ctx: &Context) -> Ex {
    ctx.symbol_with("x", &[Assumption::Real]).unwrap()
}

/// `e` at `x = r` is zero: structurally, or below `1e-12` relative to the
/// terms at 16 digits.
fn assert_root(e: &Ex, x: &Ex, r: &Ex) {
    let at = e.subs(x, r).eval();
    if at.is_zero_structural() {
        return;
    }
    let v = at
        .eval_complex64()
        .unwrap_or_else(|err| panic!("{e} at {r}: {at} ({err})"));
    let scale = e
        .expand()
        .args()
        .iter()
        .filter_map(|t| t.subs(x, r).eval_complex64().ok())
        .map(|z| z.norm())
        .fold(1.0f64, f64::max);
    assert!(v.norm() < 1e-12 * scale, "{e} at {r} is {v}");
}

/// The roots of `src` in `x`, each checked by substitution, as sorted
/// real values (all roots must be real here).
fn real_roots(ctx: &Context, x: &Ex, src: &str) -> Vec<f64> {
    let e = parse(ctx, src);
    let got = e
        .solve(x)
        .unwrap_or_else(|err| panic!("solve({src}): {err}"));
    let mut out = Vec::new();
    for r in &got {
        assert_root(&e, x, r);
        let z = r.eval_complex64().unwrap();
        assert!(z.im.abs() < 1e-12 * (1.0 + z.re.abs()), "{src}: {r}");
        out.push(z.re);
    }
    out.sort_by(f64::total_cmp);
    out
}

fn assert_values(src: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{src}: {got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() < 1e-12 * (1.0 + w.abs()),
            "{src}: {got:?} vs {want:?}"
        );
    }
}

/// Membership of `x0` in `set` agrees with the sign of `f` there (`f > 0`
/// for `gt`).
fn assert_member(set: &SetEx, ctx: &Context, x0: &str, want: bool) {
    let p = parse(ctx, x0);
    // never the wrong answer (an undecided membership is allowed)
    assert_ne!(set.contains(&p), Some(!want), "{x0} in {set}");
}

/// Periodic inequalities were refused ("cannot build a sign chart for
/// sin(x)").  Now one period `[0, T)` is decided between the critical
/// points (families of zeros and poles from the general solver) and the
/// runs repeat with the period, reduced to the smallest one the pattern
/// has (`tan x > 1`, `sin²x > 1/4`: `π`), each run with its middle in
/// `(−T/2, T/2]`.  Oracle: the sign of the expression at the probe points
/// (`sin(13π/6 + 1/10) > 1/2`, `sin(2) > 1/2`, `sin(3) < 1/2`, …).  SymPy
/// 1.14 is wrong on every one: `solveset(sin(x) > 1/2, x, S.Reals)` →
/// `Interval.open(pi/6, 5*pi/6)` (one period), `solveset(cos(x) >= 0, x,
/// S.Reals)` → `[0, pi/2] ∪ [3*pi/2, 2*pi)`, `solveset(tan(x) > 1, x,
/// S.Reals)` → `Interval.open(pi/4, pi/2)`.
#[test]
fn periodic_inequalities_hold_in_every_period() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let s = parse(&ctx, "sin(x) - 1/2").solve_gt(&x);
    assert_eq!(
        s.to_string(),
        "ImageSet(Lambda(_n, (2*_n*pi + 1/6*pi, 2*_n*pi + 5/6*pi)), Integers)"
    );
    for (p, want) in [
        ("2", true),
        ("13/6*pi + 1/10", true),
        ("-11/6*pi + 1/10", true),
        ("3", false),
        ("1/6*pi", false),
        ("0", false),
    ] {
        assert_member(&s, &ctx, p, want);
    }
    let s = parse(&ctx, "cos(x)").solve_ge(&x);
    assert_eq!(
        s.to_string(),
        "ImageSet(Lambda(_n, [2*_n*pi - 1/2*pi, 2*_n*pi + 1/2*pi]), Integers)"
    );
    assert_member(&s, &ctx, "5/2*pi", true);
    assert_member(&s, &ctx, "pi", false);
    let s = parse(&ctx, "tan(x) - 1").solve_gt(&x);
    assert_eq!(
        s.to_string(),
        "ImageSet(Lambda(_n, (_n*pi + 1/4*pi, _n*pi + 1/2*pi)), Integers)"
    );
    // the pole is not a member
    assert_member(&s, &ctx, "3/2*pi", false);
    assert_member(&s, &ctx, "5/4*pi + 1/10", true);
    let s = parse(&ctx, "sin(x)^2 - 1/4").solve_gt(&x);
    assert_eq!(
        s.to_string(),
        "ImageSet(Lambda(_n, (_n*pi + 1/6*pi, _n*pi + 5/6*pi)), Integers)"
    );
    // sin(2x) − sin(x) = sin x·(2 cos x − 1): two runs per period
    let s = parse(&ctx, "sin(2*x) - sin(x)").solve_gt(&x);
    for (p, want) in [("1/2", true), ("-2", true), ("2", false), ("-1/2", false)] {
        assert_member(&s, &ctx, p, want);
    }
    assert_eq!(
        parse(&ctx, "sin(x) - 2").solve_gt(&x).to_string(),
        "EmptySet"
    );
    assert_eq!(
        parse(&ctx, "sin(x) + 2").solve_gt(&x).to_string(),
        "(-oo, oo)"
    );
    // sin x ≥ 1 holds at the points 2nπ + π/2 only
    assert_eq!(
        parse(&ctx, "sin(x) - 1").solve_ge(&x).to_string(),
        "ImageSet(Lambda(_n, 2*_n*pi + 1/2*pi), Integers)"
    );
    // with a bounded window the families are listed
    let c = ctx.parse_bool("sin(x) > 1/2 & x > 0 & x < 10").unwrap();
    assert_eq!(
        reduce_inequalities(&[c], &x).unwrap().to_string(),
        "(1/6*pi, 5/6*pi) ∪ (13/6*pi, 17/6*pi)"
    );
}

/// The real solutions of an equation with a parameter came back without
/// the real line: `x² + a = 0` gave `{√(−a), −√(−a)}`, imaginary for
/// `a > 0`, and `eˣ + 3 = y` gave `{ln(y − 3)}`, not real for `y < 3`.
/// Members not known to be real now keep `∩ ℝ`, and members that are
/// certainly not real are dropped.  SymPy: `solveset(x**2 + a, x, S.Reals)`
/// → `Intersection({-sqrt(-a), sqrt(-a)}, Reals)`.
#[test]
fn real_solution_sets_keep_the_line_for_undecided_members() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let _ = ctx.symbol("a");
    let s = parse(&ctx, "x^2 + a").solve_as_set(&x);
    assert_eq!(s.to_string(), "(-oo, oo) ∩ {sqrt(-a), -sqrt(-a)}");
    let _ = ctx.symbol_with("y", &[Assumption::Positive]).unwrap();
    let s = parse(&ctx, "exp(x) + 3 - y").solve_as_set(&x);
    assert_eq!(s.to_string(), "(-oo, oo) ∩ {ln(y - 3)}");
    // known real members need no line, certainly non-real ones go
    assert_eq!(
        parse(&ctx, "x^2 - 4").solve_as_set(&x).to_string(),
        "{-2, 2}"
    );
    assert_eq!(
        parse(&ctx, "x^2 + 4").solve_as_set(&x).to_string(),
        "EmptySet"
    );
}

/// For a positive `x`, `x − atan(1/2) − atan(1/3) + π/4 = 0` (its root is
/// `0`: `atan(1/2) + atan(1/3) = π/4`) had the solution set `(0, ∞) ∩
/// {atan(1/2) + atan(1/3) − π/4}`: `solve_as_set` did not use the exact zero
/// test that `solve` used.  Oracle: `tan(atan(1/2) + atan(1/3)) = (1/2 +
/// 1/3)/(1 − 1/6) = 1`.
#[test]
fn solution_sets_decide_members_that_are_exactly_zero() {
    let ctx = Context::new();
    let pos = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let e = parse(&ctx, "x - atan(1/2) - atan(1/3) + pi/4");
    assert_eq!(e.solve_as_set(&pos).to_string(), "EmptySet");
    let nonneg = {
        let ctx2 = Context::new();
        let x = ctx2.symbol_with("x", &[Assumption::NonNegative]).unwrap();
        parse(&ctx2, "x - atan(1/2) - atan(1/3) + pi/4")
            .solve_as_set(&x)
            .to_string()
    };
    assert_eq!(nonneg, "{0}");
}

/// An inequality whose expression has non-real coefficients holds where
/// it is real: `((x − 1)(x − 2) + i)·((x − 1)(x − 2) + 2i) < 0` is `−2 < 0`
/// at `x = 1` and `x = 2` and not real elsewhere; it was `EmptySet` (the
/// sign chart sampled between the zeros of the expression, all complex).
/// SymPy: `solve_univariate_inequality(…, x, relational=False)` → `{1, 2}`.
#[test]
fn complex_valued_inequality_holds_where_it_is_real() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let e = parse(&ctx, "((x - 1)*(x - 2) + I)*((x - 1)*(x - 2) + 2*I)");
    assert_eq!(e.solve_lt(&x).to_string(), "{1, 2}");
    assert_eq!(e.solve_gt(&x).to_string(), "EmptySet");
}

/// The two sides of an inequality are defined only where both are:
/// `x + 1/x > −2 + 1/x` was `(−2, ∞)`, the pole `0` included (`lhs − rhs`
/// cancels `1/x`).  SymPy: `solveset(x + 1/x > -2 + 1/x, x, S.Reals)` →
/// `Interval.open(-2, 0) ∪ Interval.open(0, oo)`.  Bounds at `±∞` were
/// "condition does not involve the variable": `−∞ < x` is `ℝ`, and `√x < ∞`
/// is the domain `[0, ∞)` of `√x`.
#[test]
fn inequality_sides_keep_their_poles_and_infinite_bounds_decide() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let red = |src: &str| {
        let c = ctx.parse_bool(src).unwrap();
        reduce_inequalities(&[c], &x).unwrap().to_string()
    };
    assert_eq!(red("x + 1/x > -2 + 1/x"), "(-2, 0) ∪ (0, oo)");
    assert_eq!(red("-oo < x"), "(-oo, oo)");
    assert_eq!(red("x >= 0 & -oo < x & x < oo"), "[0, oo)");
    assert_eq!(red("sqrt(x) < oo"), "[0, oo)");
    assert_eq!(red("x > oo"), "EmptySet");
}

/// A periodic function of a periodic family (`sin(cos x) = 1/2`: `cos x =
/// π/6 + 2nπ`, then `x = ±acos(π/6 + 2nπ) + 2mπ`) took the same parameter
/// for both layers: for a real `x` the set was `±acos(2nπ + π/6) + 2nπ` cut
/// to the line, i.e. `±acos(π/6)` alone, without `±acos(π/6) + 2mπ`.  Now
/// the members of the inner family in the range of `cos` are listed, each
/// with a family of its own; where two independent parameters would be
/// needed (`sin(eˣ) = 1/2` over ℂ: `2nπ + ln(2mπ + π/6)`) the set is not
/// known (`ConditionSet`; it was the subset with `m = n`).  Oracle: the
/// members substituted (`sin(cos(acos(π/6) + 2π)) = 1/2`); SymPy:
/// `ConditionSet(x, Eq(sin(cos(x)), 1/2), Reals)` (refused).
#[test]
fn nested_periodic_functions_take_independent_periods() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let e = parse(&ctx, "sin(cos(x)) - 1/2");
    let s = e.solve_as_set(&x);
    assert_eq!(
        s.to_string(),
        "ImageSet(Lambda(_n, 2*_n*pi - acos(1/6*pi)), Integers) ∪ ImageSet(Lambda(_n, 2*_n*pi + acos(1/6*pi)), Integers)"
    );
    for m in ["acos(pi/6) + 2*pi", "-acos(pi/6) - 6*pi"] {
        let p = parse(&ctx, m);
        assert_root(&e, &x, &p);
        assert_eq!(s.contains(&p), Some(true), "{m}");
    }
    let plain = ctx.symbol("z");
    let s = parse(&ctx, "sin(exp(z)) - 1/2").solve_as_set(&plain);
    assert!(s.to_string().starts_with("ConditionSet"), "{s}");
}

/// A periodic equation in the unknown of a substitution kept its principal
/// roots only: for a real `x`, `cot(eˣ) = 2` had the solution set
/// `{ln(2·atan(√5 − 2))}`, a claim that there is one solution, while
/// `ln(acot 2 + nπ)` solves it for every `n ≥ 0`.  Oracle: the members
/// substituted, and `ln(acot 2 + nπ)` for `n = 0…6` (−0.76862…, 1.28239…,
/// 1.90907…, …, 2.9606…) the scan roots of `cot(eˣ) − 2` on `[−1, 3]`;
/// SymPy: `ImageSet(Lambda(_n, log(_n*pi + acot(2))), Integers)`.
#[test]
fn periodic_equation_in_a_substituted_unknown_keeps_its_families() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let e = parse(&ctx, "cot(exp(x)) - 2");
    let s = e.solve_as_set(&x);
    assert!(s.to_string().contains("ImageSet"), "{s}");
    // ln(acot 2 + nπ), n = 0 … 6: roots and members (never non-members)
    for n in 0..=6 {
        let m = parse(&ctx, &format!("ln(acot(2) + {n}*pi)"));
        assert_root(&e, &x, &m);
        assert_ne!(s.contains(&m), Some(false), "n = {n} in {s}");
    }
}

/// The real solutions of a Lambert equation with a parameter: `a/x +
/// e^(x/2) = 0` gave `{2·W(−a/2)}`, missing `2·W₋₁(−a/2)` (for `0 < a < 2/e`
/// both are real) and not real for `a > 2/e`; `e^(x/2)/4 + 2a/x³ = 0` gave
/// `{6·W(∛(−8a)/6)}` with the principal cube root, not real for `a > 0`,
/// which misses both real roots `6·W_k(−∛a/3)`.  Now both branches and all
/// cube roots, cut to the line.  Oracle: at `a = 1/10` the members that are
/// real solve the equation and match the real-root scan of `e^(x/2)/4 +
/// 1/(5x³)` (sign changes at −7.7908…, −4.5072…: mpmath `findroot`); SymPy
/// gives `ConditionSet`s (XFAIL `test_solve_lambert`).
#[test]
fn parametric_lambert_sets_take_every_real_branch() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let a = ctx.symbol_with("a", &[Assumption::Real]).unwrap();
    let s = parse(&ctx, "a/x + exp(x/2)").solve_as_set(&x);
    assert_eq!(s.to_string(), "(-oo, oo) ∩ {2*W(-1/2*a, -1), 2*W(-1/2*a)}");
    let s = parse(&ctx, "exp(x/2)/4 + 2*a/x^3")
        .solve_as_set(&x)
        .to_string();
    // three cube roots, two branches each, cut to the line
    assert!(s.starts_with("(-oo, oo) ∩ {"), "{s}");
    assert_eq!(s.matches("W(").count(), 6, "{s}");
    let _ = a;
}

/// One value written in several forms is one root: for a positive `x`,
/// `x^(−1/3)·(231361/10000 + π²/x²)/6 = π²/(2·x^(7/3))` gave `100√2·π/481`
/// and two radical forms of the same number (`t³` for the roots `t` of the
/// equation in `t = x^(1/3)`).  Oracle: the forms agree to 30 digits and
/// their difference divided by `π` is an algebraic 0; SymPy (XFAIL
/// `test_issue_24609_xfail`) gives three roots, two of them negative.
#[test]
fn equal_roots_in_other_forms_are_one() {
    let ctx = Context::new();
    let pos = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let src = "x^(-1/3)*(231361/10000 + pi^2/x^2)/6 - pi^2/(2*x^(7/3))";
    let got = parse(&ctx, src).solve(&pos).unwrap();
    assert_eq!(got.len(), 1, "{got:?}");
    assert_eq!(got[0].to_string(), "100/481*sqrt(2)*pi");
}

/// The resolvent cubic of Ferrari's method lost its rational root when a
/// coefficient had no divisor list (above 10⁶), and Cardano's formula
/// wrote the quartic's roots in cube roots: the Rothstein–Trager
/// polynomial `7168t⁴ + 7168t³ + 96t² + 16t + 1` of `∫ x³/(x⁴ + 4x³ − 1) dx`
/// (resolvent root `−9/224`) had roots in `∛(81/44957696·√21 ± …)`.
/// Oracle: SymPy `roots(7168*x**4 + 7168*x**3 + 96*x**2 + 16*x + 1)` →
/// `-1/4 ± 3√2/16 ± √(99/896 ∓ 9√2/112)` (in `ℚ(√2)` and square roots of it).
#[test]
fn ferrari_resolvent_with_large_coefficients_finds_its_rational_root() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let e = parse(&ctx, "7168*t^4 + 7168*t^3 + 96*t^2 + 16*t + 1");
    let got = e.solve(&t).unwrap();
    assert_eq!(got.len(), 4);
    for r in &got {
        assert_root(&e, &t, r);
        let s = r.to_string();
        assert!(!s.contains("cbrt") && !s.contains("^(1/3)"), "{s}");
    }
}

/// Exponentials of bases with no common power were refused: `5^(x/2) =
/// 2^(x/3)`, `3^(2x) = 2^(x + 3)`, `4^(x+1) + 4^(x−1) + 4^(x+2) = 3^(x+2) +
/// 3^(x+3)`, `5^(x/2) = 2^(3/x)`, `7^(x² − 80) = 49^x`.  Two groups of
/// exponentials `C₁e^(S₁) = −C₂e^(S₂)` give `S₁ − S₂ = ln(−C₂/C₁)`.  Oracle:
/// the roots substituted; SymPy `solveset(…, x, S.Reals)` → `{0}`,
/// `{-3*log(2)/(-2*log(3) + log(2))}` (1.38253626185511…), `{2}`,
/// `{±sqrt(6)*sqrt(log(2))/sqrt(log(5))}` (±1.60750096374476…), `{-8, 10}`.
#[test]
fn exponentials_of_different_bases() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    for (src, want) in [
        ("5^(x/2) - 2^(x/3)", vec![0.0]),
        ("3^(2*x) - 2^(x + 3)", vec![1.382_536_261_855_110_7]),
        ("4^(x+1) + 4^(x-1) + 4^(x+2) - 3^(x+2) - 3^(x+3)", vec![2.0]),
        (
            "5^(x/2) - 2^(3/x)",
            vec![-1.607_500_963_744_768, 1.607_500_963_744_768],
        ),
        ("7^(x^2 - 80) - 49^x", vec![-8.0, 10.0]),
        ("2^(x + 1) - 9*exp(5 - 2*x)", vec![2.415_047_140_284_394]),
    ] {
        assert_values(src, &real_roots(&ctx, &x, src), &want);
    }
    // −C₂/C₁ < 0: no real solution
    assert!(matches!(
        parse(&ctx, "2^x + 3^x").solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
}

/// Equations with several absolute values were refused; for a real `x`
/// each choice of signs is a polynomial equation, and a candidate the
/// signs at it contradict fails the check.  Oracle: SymPy `solve(Abs(x + 3)
/// - 2*Abs(x - 3), x)` → `[1, 9]`, `solve(Abs(x + 1) + Abs(x**2 - 7) - 9)` →
/// `[-1/2 + sqrt(61)/2, 1/2 - sqrt(69)/2]`, `solve(2*Abs(x) - Abs(x - 1))`
/// → `[-1, 1/3]`.  An imaginary unknown is not split (`|i·t| = |t|`).
#[test]
fn several_absolute_values() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let s61 = 61f64.sqrt();
    let s69 = 69f64.sqrt();
    for (src, want) in [
        ("abs(x + 3) - 2*abs(x - 3)", vec![1.0, 9.0]),
        (
            "abs(x + 1) + abs(x^2 - 7) - 9",
            vec![(1.0 - s69) / 2.0, (s61 - 1.0) / 2.0],
        ),
        ("2*abs(x) - abs(x - 1)", vec![-1.0, 1.0 / 3.0]),
    ] {
        assert_values(src, &real_roots(&ctx, &x, src), &want);
    }
    let i = ctx.symbol_with("i", &[Assumption::Imaginary]).unwrap();
    assert!(parse(&ctx, "abs(i) - 3").solve(&i).is_err());
}

/// Inversions that were missing: `erf`, `erfc` and their inverses,
/// `W(f) = y`, `atan2(f, c) = y` for `c > 0`, `f^π = c`, a rational function
/// under a root or a logarithm (`√((x − 3)/x) = 1/2`, `ln((x + 1)(x − 1)) =
/// 0`), `C(x, k)` as a polynomial.  Oracle: SymPy `solve(erf(x) - 1/2)` →
/// `[erfinv(1/2)]`, `solve(LambertW(2*x) - y, x)` → `[y*exp(y)/2]`,
/// `solve(atan2(x, 2) - pi/3)` → `[2*sqrt(3)]`, `solve(x**pi - 2)` →
/// `[2**(1/pi)]`, `solveset(sqrt((x - 3)/x) - 1/2, x, S.Reals)` → `{4}`,
/// `solveset(log((x + 1)*(x - 1)), x, S.Reals)` → `{-sqrt(2), sqrt(2)}`,
/// `solve(binomial(x, 2) - 3)` → `[-2, 3]`; `erf(x) = 1` has no real
/// solution (`|erf| < 1`).
#[test]
fn inversions_of_erf_lambert_atan2_powers_and_binomials() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let show = |src: &str| -> Vec<String> {
        let mut v: Vec<String> = parse(&ctx, src)
            .solve(&x)
            .unwrap_or_else(|e| panic!("{src}: {e}"))
            .iter()
            .map(|r| r.to_string())
            .collect();
        v.sort();
        v
    };
    assert_eq!(show("erf(x) - 1/2"), ["erfinv(1/2)"]);
    assert_eq!(show("erfinv(x) - 2"), ["erf(2)"]);
    assert_eq!(show("atan2(x, 2) - pi/3"), ["2*sqrt(3)"]);
    assert_eq!(show("x^pi - 2"), ["2^(1/pi)"]);
    assert_eq!(show("binomial(x, 2) - 3"), ["-2", "3"]);
    assert_eq!(
        parse(&ctx, "sqrt((x - 3)/x) - 1/2")
            .solve_as_set(&x)
            .to_string(),
        "{4}"
    );
    assert_eq!(
        parse(&ctx, "log((x + 1)*(x - 1))")
            .solve_as_set(&x)
            .to_string(),
        "{sqrt(2), -sqrt(2)}"
    );
    assert!(matches!(
        parse(&ctx, "erf(x) - 1").solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
    let y = ctx.symbol("y");
    let t = ctx.symbol("t");
    let got = parse(&ctx, "LambertW(2*t) - y").solve(&t).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].to_string(), "1/2*y*exp(y)");
    let _ = y;
}

/// The poles inside a factor came back as roots: `d·(1 + 1/d) = 0` with
/// `d = x⁵ − x + 1` gave the five roots of `d` (where it is `0·∞`) besides
/// those of `d + 1`.  Oracle: SymPy `solve(d*(1 + 1/d))` →
/// `[CRootOf(x**5 - x + 2, k) …]`; at a root of `d` the expression is
/// undefined.
#[test]
fn poles_inside_a_factor_are_not_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let got = parse(&ctx, "(x^5 - x + 1)*(1 + 1/(x^5 - x + 1))")
        .solve(&x)
        .unwrap();
    assert_eq!(got.len(), 5, "{got:?}");
    for r in &got {
        assert!(r.to_string().starts_with("RootOf(x^5 - x + 2"), "{r}");
    }
}

/// Three or four square roots of polynomials were refused: the norm over
/// every choice of signs is a polynomial whose roots are the candidates;
/// a norm that is a nonzero constant proves there is no root (`x + √(x² −
/// 5) = 0`: the norm is 5), which also decides the sign chart of `x +
/// √(x² − 5) > 0`.  Oracle: SymPy `solve(sqrt(2*x + 9) - sqrt(x + 1) -
/// sqrt(x + 4))` → `[0]`, `solve(x + sqrt(x**2 - 5))` → `[]`,
/// `solveset(x + sqrt(x**2 - 5) > 0, x, S.Reals)` → `Interval(sqrt(5), oo)`.
#[test]
fn several_square_roots() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    assert_values(
        "sqrt(2x + 9) = sqrt(x + 1) + sqrt(x + 4)",
        &real_roots(&ctx, &x, "sqrt(2*x + 9) - sqrt(x + 1) - sqrt(x + 4)"),
        &[0.0],
    );
    assert!(matches!(
        parse(&ctx, "x + sqrt(x^2 - 5)").solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
    assert_eq!(
        parse(&ctx, "x + sqrt(x^2 - 5)").solve_gt(&x).to_string(),
        "[sqrt(5), oo)"
    );
}

/// Trigonometric functions of multiples of `x` with a common factor were
/// refused (`sin 600x` has a half-angle polynomial of degree 1200): in
/// `u = r·x` the multiples are coprime.  Oracle: the members substituted;
/// SymPy `solveset(sin(300*x) + sin(600*x), x, S.Reals)` → the families
/// `nπ/150 + {0, π/450, π/300, π/225}` (the same set), and
/// `solveset(cos(x/5) + cos(x/15), x, S.Reals)` → six families of period
/// `30π` (the same set).
#[test]
fn trigonometric_multiples_with_a_common_factor() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    assert_eq!(
        parse(&ctx, "sin(300*x) + sin(600*x)")
            .solve_as_set(&x)
            .to_string(),
        "ImageSet(Lambda(_n, 1/300*_n*pi), Integers) ∪ ImageSet(Lambda(_n, 1/450*_n*pi), Integers)"
    );
    let e = parse(&ctx, "cos(x/5) + cos(x/15)");
    let s = e.solve_as_set(&x);
    assert_eq!(
        s.to_string(),
        "ImageSet(Lambda(_n, 15*_n*pi + 15/2*pi), Integers) ∪ ImageSet(Lambda(_n, 15/2*_n*pi + 15/4*pi), Integers)"
    );
    for m in ["75/4*pi", "45/2*pi", "105/4*pi"] {
        let p = parse(&ctx, m);
        assert_root(&e, &x, &p);
        assert_eq!(s.contains(&p), Some(true), "{m}");
    }
}

/// The poles of `tan u` are roots only where the expression tends to 0 as
/// `tan u → ∞` (`1/tan x`): after the members of `cos(4·sin x) = 0` were
/// listed, the families `±asin(π/8) + 2nπ` of `tan(4·sin x) = −√2/2` (poles
/// of `tan`, where it is infinite) passed the check.  Oracle: the six
/// families `4·sin x = atan(−√2/2) + kπ`, `k = −1, 0, 1`, substituted; SymPy
/// `ConditionSet` (refused).
#[test]
fn tangent_poles_are_roots_only_where_the_expression_vanishes() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let e = parse(&ctx, "tan(4*sin(x)) + sqrt(2)/2");
    let s = e.solve_as_set(&x);
    assert!(!s.to_string().contains("asin(-1/8*pi)"), "{s}");
    let window = ctx.interval(&ctx.int(-4), &ctx.int(4), IntervalKind::Closed);
    let members = s.intersection(&window).simplify().as_finite_set().unwrap();
    // two per family in [−4, 4] for the k = 0 families, …: every member a root
    assert!(members.len() >= 6, "{members:?}");
    for m in &members {
        assert_root(&e, &x, m);
    }
    assert_eq!(
        parse(&ctx, "1/tan(x)").solve_as_set(&x).to_string(),
        "ImageSet(Lambda(_n, _n*pi + 1/2*pi), Integers)"
    );
}

/// A system without unknowns: the empty assignment solves it only when
/// every equation holds.  Up to 0.46 `solve_system_ex([42], [])` was
/// `[[]]` (one solution) — SymPy 1.14 test_solvers: `solve(42)` is `[]`.
#[test]
fn constant_systems_without_unknowns() {
    let ctx = Context::new();
    let none: Vec<Ex> = Vec::new();
    let r = symplex::polysys::solve_system_ex(&[ctx.int(42)], &none).unwrap();
    assert!(r.is_empty(), "{r:?}");
    let r = symplex::polysys::solve_system_ex(&[ctx.int(1), ctx.int(2)], &none).unwrap();
    assert!(r.is_empty(), "{r:?}");
    let r = symplex::polysys::solve_system_ex(&[ctx.int(0)], &none).unwrap();
    assert_eq!(r.len(), 1, "{r:?}");
}
