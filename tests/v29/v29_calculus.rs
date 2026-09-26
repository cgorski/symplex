//! After 0.29 — the calculus differential hunter: `integrate_definite`,
//! `limit`, `series`, `summation`, `solve` and `diff` checked against
//! independent numerical oracles (an adaptive quadrature over an own `f64`
//! evaluator, certified `eval_decimal` at exact rational points, partial
//! sums), and the unbounded exact integer sequences of `eval`.
//!
//! Each test says what was wrong before; every reference value cites the
//! oracle call that produced it (mpmath 1.3.0 at `mp.dps = 30`, SymPy 1.14).

use std::time::{Duration, Instant};

use symplex::prelude::*;

/// `|a − b| ≤ tol·max(1, |b|)` for the value of `e` against `expected`.
fn assert_close(e: &Ex, expected: f64, tol: f64, what: &str) {
    let v = e
        .eval_f64()
        .unwrap_or_else(|err| panic!("{what}: {e} does not evaluate: {err}"));
    assert!(
        (v - expected).abs() <= tol * expected.abs().max(1.0),
        "{what}: {e} = {v}, expected {expected}"
    );
}

fn limit_dir(ctx: &Context, f: &str, p: &str, dir: Direction) -> Result<Ex, SymplexError> {
    let x = ctx.symbol("x");
    let f = ctx.parse(f).unwrap();
    let p = ctx.parse(p).unwrap();
    f.try_limit_dir(&x, &p, dir)
}

// ═══════════════════════════════════════════════════════════════════════════
// Zero constants that canonicalisation does not see
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn a_hidden_zero_leading_coefficient_is_not_a_leading_term() {
    // Gruntz rewrites asinh(2 + 1/w) into ln(2 + 1/w + √(…)) but leaves the
    // constant asinh(2); the leading coefficient asinh(2) − ln(2 + √5) is 0
    // without cancelling, and the limit came out 0.
    // mpmath: limit(lambda x: x/(asinh(x+2)-asinh(2)), 0) = 2.23606797749978969640917366873
    let ctx = Context::new();
    for dir in [Direction::Left, Direction::Right, Direction::Both] {
        let l = limit_dir(&ctx, "x/(asinh(x + 2) - asinh(2))", "0", dir).unwrap();
        assert_close(
            &l,
            2.236_067_977_499_79,
            1e-12,
            "lim x/(asinh(x+2) − asinh 2)",
        );
    }
    // The same hidden zero written out: direct substitution folded
    // 0·(1/0) to 0.
    let l = limit_dir(
        &ctx,
        "x/(asinh(x + 2) - ln(2 + sqrt(5)))",
        "0",
        Direction::Both,
    )
    .unwrap();
    assert_close(
        &l,
        2.236_067_977_499_79,
        1e-12,
        "lim x/(asinh(x+2) − ln(2+√5))",
    );
    // mpmath: limit(lambda t: (t-1)/(tan(exp(t))-tan(e)), 1) = 0.305802997566294853439927024316
    let l = limit_dir(&ctx, "(x - 1)/(tan(exp(x)) - tan(E))", "1", Direction::Both).unwrap();
    assert_close(
        &l,
        0.305_802_997_566_294_85,
        1e-12,
        "lim (x−1)/(tan eˣ − tan e)",
    );
}

#[test]
fn a_hidden_zero_series_coefficient_is_zero() {
    // The series engine took the constant term asinh(2) − ln(2 + √5) of the
    // denominator for non-zero: the series was x/(asinh 2 − ln(2 + √5)).
    // SymPy 1.14 gives the same wrong `x/(-log(2 + sqrt(5)) + asinh(2)) + O(x**2)`;
    // with the zero visible, `series(x/(asinh(x+2)-asinh(2)), x, 0, 2)` =
    // `sqrt(5) + sqrt(5)*x/5 + O(x**2)`.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("x/(asinh(x + 2) - ln(2 + sqrt(5)))").unwrap();
    let s = f.series(&x, &ctx.int(0), 2);
    let expected = ctx.parse("sqrt(5) + sqrt(5)*x/5").unwrap();
    assert!(
        (&s - &expected).simplify().is_zero_structural(),
        "series = {s}"
    );
    // 1/(x + sin²1 + cos²1 − 1) = 1/x.
    let f = ctx.parse("1/(x + sin(1)^2 + cos(1)^2 - 1)").unwrap();
    assert_eq!(f.series(&x, &ctx.int(0), 2).to_string(), "1/x");
}

// ═══════════════════════════════════════════════════════════════════════════
// Limits
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn the_sign_of_a_non_real_constant_is_not_positive() {
    // cosh and exp were positive whatever their argument: cosh(acos 4) =
    // cos(acosh 4) is negative, and the limit was +∞.
    // mpmath: cosh(acos(4)) = -0.472954258031328548472382981861
    let ctx = Context::new();
    let l = limit_dir(
        &ctx,
        "(asinh(x) - 2*x)*cosh(acos(4))",
        "-oo",
        Direction::Both,
    )
    .unwrap();
    assert_eq!(l.to_string(), "-oo");
}

#[test]
fn zero_to_a_vanishing_power() {
    // 0^g became exp(g·ln 0) with ln 0 taken for a finite coefficient:
    // lim_{x→0⁺} 0^x was 1.  0^x = 0 for x > 0 and is complex infinity
    // for x < 0.
    let ctx = Context::new();
    assert_eq!(
        limit_dir(&ctx, "0^x", "0", Direction::Right)
            .unwrap()
            .to_string(),
        "0"
    );
    assert!(limit_dir(&ctx, "0^x", "0", Direction::Left).is_err());
    assert!(limit_dir(&ctx, "0^(x - 1)", "1", Direction::Both).is_err());
}

#[test]
fn the_sign_of_a_complex_function_is_not_taken_from_its_limit() {
    // (−2)^(−x) tends to 1 but is not real; |·| was dropped as if it were
    // positive and lim_{x→0⁺} |(−2)^(−x)|^(1/x) came out −1/2.  For x > 0,
    // |(−2)^(−x)| = 2^(−x), so the limit is 1/2.
    let ctx = Context::new();
    // Not computed is acceptable; a value must be 1/2, never −1/2.
    if let Ok(l) = limit_dir(&ctx, "abs((-2)^(-x))^(1/x)", "0", Direction::Right) {
        assert_close(&l, 0.5, 1e-12, "lim |(−2)^(−x)|^(1/x)");
    }
}

#[test]
fn irrational_exponents_order_leading_terms() {
    // Gruntz keyed a non-rational exponent (−π, −√2) as 0.
    let ctx = Context::new();
    let oo = |f: &str| {
        limit_dir(&ctx, f, "oo", Direction::Both)
            .unwrap()
            .to_string()
    };
    assert_eq!(oo("x^sqrt(2) - x^(7/5)"), "oo"); // was −∞ (√2 > 7/5)
    assert_eq!(oo("(x^pi + x)/x^pi"), "1"); // was 0
    assert_eq!(oo("(x^pi - ln(x))/cos(3/2)"), "oo"); // was −∞; cos(3/2) > 0
}

#[test]
fn exp_of_a_logarithm_is_a_power_for_gruntz() {
    // sinh(ln x) rewritten into exponentials left exp(−ln(1/x)) in the MRV
    // set beside x: lim_{x→0} x·sinh(ln x) was 0.
    // mpmath: limit(lambda t: t*sinh(log(t)), 0) = -0.5
    let ctx = Context::new();
    let l = limit_dir(&ctx, "x*sinh(ln(x))", "0", Direction::Right).unwrap();
    assert_eq!(l.to_string(), "-1/2");
    let l = limit_dir(&ctx, "cosh(ln(x))/x", "oo", Direction::Both).unwrap();
    assert_eq!(l.to_string(), "1/2");
}

#[test]
fn a_limit_on_a_branch_cut_depends_on_the_side() {
    // atanh(1 − x) → iπ/2 from Re < 0 as x → ∞; the composition rule took
    // asinh at iπ/2, on its cut, and returned asinh(iπ/2) = acosh(π/2) + iπ/2.
    // mpmath: asinh(atanh(1-mpf(10)**30)) =
    //   (-1.02322747854755057931749567795 + 1.57079632679489661923132169164j)
    let ctx = Context::new();
    if let Ok(l) = limit_dir(&ctx, "asinh(atanh(1 - x))", "oo", Direction::Right) {
        let z = l.eval_complex64().unwrap();
        assert!(
            (z.re + 1.023_227_478_547_550_6).abs() < 1e-12
                && (z.im - std::f64::consts::FRAC_PI_2).abs() < 1e-12,
            "lim asinh(atanh(1 − x)) = {l} = {z}"
        );
    }
    // A real argument moving along a cut is fine.
    let l = limit_dir(&ctx, "sqrt(x - 2)", "0", Direction::Right).unwrap();
    assert_eq!(l.to_string(), "sqrt(2)*I");
}

// ═══════════════════════════════════════════════════════════════════════════
// Series
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sign_and_heaviside_expand_by_the_sign_of_their_argument() {
    // sign(g) went to the differentiation fallback: sign(g(0)) = sign(0) = 0
    // and d sign = 0, so atanh(x²)·sign(x²) expanded to 0 (it is x² + x⁶/3 + …),
    // and tan(sign x) — no two-sided expansion — to 0.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = |f: &str, n: u32| ctx.parse(f).unwrap().series(&x, &ctx.int(0), n);
    assert_eq!(s("atanh(x^2)*sign(x^2)", 5).to_string(), "x^2");
    assert!(s("tan(sign(x))", 6).has_unevaluated());
    assert_eq!(s("x*Heaviside(x^2)", 3).to_string(), "x");
}

#[test]
fn the_differentiation_fallback_does_not_fold_singular_products() {
    // The value at the point was subs + eval, which folds 0·sinh(ln 0) to 0:
    // x·sinh(ln x) expanded to x²/2 instead of (x² − 1)/2, and sinh(ln x)
    // lost its pole.  SymPy 1.14: `simplify(sinh(log(x)).rewrite(exp))` =
    // `(x**2 - 1)/(2*x)`.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx.parse("sinh(ln(x))").unwrap().series(&x, &ctx.int(0), 3);
    let expected = ctx.parse("x/2 - 1/(2*x)").unwrap();
    assert!(
        (&s - &expected).simplify().is_zero_structural(),
        "series = {s}"
    );
}

#[test]
fn branch_point_series_terminate() {
    // Pathological fallbacks (the limit of each derivative, expanded) ran
    // for minutes: abs(ln(2 + x)) at 1 (a regression of the derivative of
    // |f| during this work), atanh(acosh x)³ at −1, acosh((x − 1/2)·sign x) at 0.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let start = Instant::now();
    let s = ctx
        .parse("abs(ln(2 + x))")
        .unwrap()
        .series(&x, &ctx.int(1), 4);
    // ln(x + 3) around x = 0 is ln 3 + x/3 − x²/18 + x³/81 (ln 3 > 0).
    let expected = ctx
        .parse("ln(3) + (x - 1)/3 - (x - 1)^2/18 + (x - 1)^3/81")
        .unwrap();
    assert!(
        (&s - &expected).expand().simplify().is_zero_structural(),
        "series = {s}"
    );
    let _ = ctx
        .parse("atanh(acosh(x))^3")
        .unwrap()
        .series(&x, &ctx.int(-1), 2);
    let _ = ctx
        .parse("acosh((x - 1/2)*sign(x))")
        .unwrap()
        .series(&x, &ctx.int(0), 5);
    assert!(
        start.elapsed() < Duration::from_secs(60),
        "took {:?}",
        start.elapsed()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// diff
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn the_derivative_of_abs_of_a_complex_function_is_real() {
    // d|f|/dx = sign(f)·f' holds for a real f only: for f complex at real x
    // the derivative of the real function |f| came out complex or with the
    // wrong sign.  Now re(conj(f)·f')/|f|.
    // mpmath: diff(lambda t: fabs(sinh(t) - mp.power(-1, mpf(-1)/3)), mpf(1)/3)
    //   = -0.192397150627760643447530949987
    // mpmath: diff(lambda t: fabs(1/(2*sqrt(t))), mpf(-1)/2) = 0.707106781186547524400844362105
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d = ctx.parse("abs(sinh(x) - (-1)^(-1/3))").unwrap().diff(&x);
    let v = d.subs(&x, &ctx.rational(1, 3)).eval_complex64().unwrap();
    assert!(
        (v.re + 0.192_397_150_627_760_64).abs() < 1e-12 && v.im.abs() < 1e-12,
        "{d} = {v}"
    );
    let d = ctx.parse("abs(1/(2*sqrt(x)))").unwrap().diff(&x);
    assert_close(
        &d.subs(&x, &ctx.rational(-1, 2)),
        0.707_106_781_186_547_5,
        1e-12,
        "d|1/(2√x)|",
    );
    // Real arguments keep sign(f)·f'.
    assert_eq!(
        ctx.parse("abs(x^2 - 1)").unwrap().diff(&x).to_string(),
        "2*x*sign(x^2 - 1)"
    );
    assert_eq!(ctx.parse("abs(x)").unwrap().diff(&x).to_string(), "sign(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// solve
// ═══════════════════════════════════════════════════════════════════════════

fn solve_str(ctx: &Context, f: &str) -> Result<Vec<Ex>, SymplexError> {
    let x = ctx.symbol("x");
    ctx.parse(f).unwrap().solve(&x)
}

#[test]
fn principal_branches_that_miss_the_right_hand_side_give_no_solution() {
    // Peeling returned a solution whenever it could invert: asin(2x) = π
    // gave 0, 1/√x = −2 gave 1/4, x^(1/3) = −3/4 gave −27/64, atan x = 2
    // gave tan 2, ln(atan x) = 1/2 gave tan(e^(1/2)), ln x = 2πi gave 1.
    let ctx = Context::new();
    for f in [
        "asin(2*x) - pi",
        "1/sqrt(x) + 2",
        "x^(1/3) + 3/4",
        "atan(x) - 2",
        "ln(atan(x)) - 1/2",
        "ln(x) - 2*pi*I",
        "acosh(x) + 1",
    ] {
        let r = solve_str(&ctx, f);
        assert!(
            matches!(r, Err(SymplexError::NoSolution { .. })),
            "solve({f}) = {r:?}"
        );
    }
}

#[test]
fn powers_are_inverted_to_all_their_roots() {
    // f^n = c returned c^(1/n) only (and its negative for a positive even
    // n): x^(−2) = 3 lost −1/√3, (x²)^(−3) = 1/2 lost −2^(1/6) and the four
    // complex roots, ln(x^(−2)) = 2 lost −1/e.
    let ctx = Context::new();
    let roots = |f: &str| {
        let mut v: Vec<(f64, f64)> = solve_str(&ctx, f)
            .unwrap()
            .iter()
            .map(|s| {
                let z = s.eval_complex64().unwrap();
                (z.re, z.im)
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    let r = roots("x^(-2) - 3");
    assert_eq!(r.len(), 2);
    assert!(
        (r[0].0 + 1.0 / 3f64.sqrt()).abs() < 1e-14 && (r[1].0 - 1.0 / 3f64.sqrt()).abs() < 1e-14
    );
    let r = roots("(x*x)^(-3) - 1/2");
    assert_eq!(r.len(), 6, "{r:?}");
    assert!(
        r.iter()
            .any(|z| (z.0 + 2f64.powf(1.0 / 6.0)).abs() < 1e-14 && z.1 == 0.0)
    );
    let r = roots("ln(x^(-2)) - 2");
    assert_eq!(r.len(), 2, "{r:?}");
    // x^(3/2) = −1: u³ = −1 with u = √x in the right half-plane, x = u²:
    // x = −1/2 ± (√3/2)i ((−1/2 ± (√3/2)i)^(3/2) = −1, SymPy 1.14:
    // `solve(x**Rational(3,2) + 1)` = [-1/2 - sqrt(3)*I/2, -1/2 + sqrt(3)*I/2]).
    let r = roots("x^(3/2) + 1");
    assert_eq!(r.len(), 2, "{r:?}");
    assert!(
        r.iter()
            .all(|z| (z.0 + 0.5).abs() < 1e-14 && (z.1.abs() - 3f64.sqrt() / 2.0).abs() < 1e-14)
    );
}

#[test]
fn radical_equations_in_the_variable_itself() {
    // √x = x − 2 had no strategy; the candidates of t = √x, t − t² + 2 = 0,
    // are t = 2 (x = 4) and t = −1 (x = 1, where √1 ≠ 1 − 2).
    let ctx = Context::new();
    let s = |f: &str| {
        solve_str(&ctx, f)
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    };
    assert_eq!(s("sqrt(x) - x + 2"), ["4"]);
    assert_eq!(s("x^(2/3) + x^(1/3) - 2"), ["1"]); // x = −8 is not a root
    assert_eq!(s("asinh(x)"), ["0"]); // no inverse of asinh before
}

// ═══════════════════════════════════════════════════════════════════════════
// Definite integrals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn kinks_the_solver_cannot_find_are_not_assumed_away() {
    // solve knew no inverse of asinh, so |asinh x| had no kink and was
    // resolved at the midpoint as asinh x on all of [−1/2, 3/2]: the result
    // was ∫ asinh = 0.8667972644….
    // mpmath: quad(lambda t: fabs(asinh(t)), [-0.5, 0, 1.5]) = 1.11194111197857618515257823139
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (a, b) = (ctx.rational(-1, 2), ctx.rational(3, 2));
    let r = ctx
        .parse("abs(asinh(x))")
        .unwrap()
        .integrate_definite(&x, &a, &b);
    assert_close(&r, 1.111_941_111_978_576_2, 1e-12, "∫|asinh x|");
    // x·eˣ = −1/5 has two real roots; solve returns W₀ only, and the result
    // was 1.3554721943… .  Now the unexplained sign change refuses the split.
    // mpmath: quad(lambda t: fabs(t*exp(t)+mpf(1)/5), [-4, lambertw(-0.2,-1), lambertw(-0.2,0), 1])
    //   = 1.56425531952357458178982315648
    let r = ctx
        .parse("abs(x*exp(x) + 1/5)")
        .unwrap()
        .integrate_definite(&x, &ctx.int(-4), &ctx.int(1));
    if !r.has_unevaluated() {
        assert_close(&r, 1.564_255_319_523_574_6, 1e-12, "∫|x eˣ + 1/5|");
    }
}

#[test]
fn a_kink_argument_changes_sign_at_its_poles() {
    // sign(1/x) flips at x = 0, a pole, not a zero: the piece [−1, 2] was
    // resolved at its midpoint as sign = 1 and ∫_{−1}^{2} x·sign(1/x) dx
    // came out 3/2.  ∫_{−1}^{0} −x dx + ∫_{0}^{2} x dx = 1/2 + 2 = 5/2.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = ctx
        .parse("x*sign(1/x)")
        .unwrap()
        .integrate_definite(&x, &ctx.int(-1), &ctx.int(2));
    assert_eq!(r.to_string(), "5/2");
    // Near-coincident kinks stay apart.
    // mpmath: quad(... fabs(t-1/3)+fabs(t-1/3-1e-13) ..., [0, 1/3, 1/3+1e-13, 1])
    //   = 0.555555555555522222222222232222
    let r = ctx
        .parse("abs(x - 1/3) + abs(x - 1/3 - 1/10^13)")
        .unwrap()
        .integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    assert_eq!(r.eval_decimal(25).unwrap(), "0.5555555555555222222222222");
}

// ═══════════════════════════════════════════════════════════════════════════
// Summation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gosper_cancels_the_polynomial_factors_of_the_term() {
    // g(k) = b(k−1)·x(k)·t(k)/c(k): c collects the polynomial factors of t,
    // and g(lo) at a common zero was 0/0.
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    // mpmath nsum(lambda k: binomial(k+1,3)/mpf(5)**k, [0, inf]) = 0.09765625;
    // SymPy: summation(binomial(k+1,3)/5**k, (k,0,oo)) = 25/256.  Was zoo.
    let t = ctx.parse("binomial(k + 1, 3)*(1/5)^k").unwrap();
    let s = t.try_summation(&k, &ctx.int(0), &ctx.infinity()).unwrap();
    assert_eq!(s.to_string(), "25/256");
    // Σ_{k=1}^{n} (2k − 2)(−1/3)^k was NaN; the partial sums (SymPy,
    // exact) are 0, 2/9, 2/27, 4/27, 28/243 for n = 1..5.
    let t = ctx.parse("(2*k - 2)*(-1/3)^k").unwrap();
    let s = t.summation(&k, &ctx.int(1), &n);
    for (m, expected) in [
        (1, "0"),
        (2, "2/9"),
        (3, "2/27"),
        (4, "4/27"),
        (5, "28/243"),
    ] {
        assert_eq!(
            s.subs_i64(&n, m).eval().to_string(),
            expected,
            "S({m}) of {s}"
        );
    }
    // (3 − n)/(n − 3) stayed in Σ_{k=0}^{n} (4 − k)(2/3)^k (NaN at n = 3).
    // SymPy: sum((4-j)*Rational(2,3)**j for j in range(4)) = 194/27.
    let t = ctx.parse("(4 - k)*(2/3)^k").unwrap();
    let s = t.summation(&k, &ctx.int(0), &n);
    assert_eq!(s.subs_i64(&n, 3).eval().to_string(), "194/27", "{s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// eval: the digit guard of the exact integer sequences
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn huge_integer_sequences_stay_symbolic() {
    // Each of these ran for more than 20 s in `eval` (release) before 0.30.
    let ctx = Context::new();
    let start = Instant::now();
    for f in [
        "fibonacci(10^7)",
        "lucas(10^7)",
        "bernoulli(10^5)",
        "factorial2(10^5)",
        "harmonic(10^6)",
        "catalan(10^6)",
        "bell(10^4)",
        "subfactorial(10^5)",
        "euler_number(10^4)",
        "stirling2(10^4, 5000)",
        "stirling1(10^4, 5000)",
        "partition_count(10^7)",
    ] {
        let e = ctx.parse(f).unwrap().eval();
        assert!(e.to_string().contains('('), "{f} evaluated to a number");
    }
    // zeta(−n) = −B_{n+1}/(n+1) went through the same recurrence.
    let _ = ctx.parse("zeta(-100001)").unwrap().eval();
    assert!(
        start.elapsed() < Duration::from_secs(20),
        "took {:?}",
        start.elapsed()
    );
}

#[test]
fn integer_sequences_inside_the_guard_are_exact() {
    // SymPy 1.14 values.
    let ctx = Context::new();
    let ev = |f: &str| ctx.parse(f).unwrap().eval().to_string();
    // bernoulli(50) = 495057205241079648212477525/66 (tangent numbers now)
    assert_eq!(ev("bernoulli(50)"), "495057205241079648212477525/66");
    // harmonic(100) (common denominator lcm(1..100) now)
    assert_eq!(
        ev("harmonic(100)"),
        "14466636279520351160221518043104131447711/2788815009188499086581352357412492142272"
    );
    // zeta(-99) = -B_100/100
    assert_eq!(
        ev("zeta(-99)"),
        "94598037819122125295227433069493721872702841533066936133385696204311395415197247711/3333000"
    );
    assert_eq!(
        ev("euler_number(40)"),
        "14851150718114980017877156781405826684425"
    );
    assert_eq!(ev("bell(30)"), "846749014511809332450147");
    assert_eq!(ev("catalan(50)"), "1978261657756160653623774456");
    assert_eq!(ev("stirling2(30, 10)"), "173373343599189364594756");
    assert_eq!(ev("stirling1(20, 5)"), "-371384787345228000");
    assert_eq!(ev("subfactorial(20)"), "895014631192902121");
    assert_eq!(ev("factorial2(31)"), "191898783962510625");
    assert_eq!(ev("lucas(100)"), "792070839848372253127");
    assert_eq!(
        ev("partition_count(1000)"),
        "24061467864032622473692149727991"
    );
    // mpmath: nstr(bernoulli(1000), 25) = -5.318704469415522036482914e+1769
    let b = ctx.parse("bernoulli(1000)").unwrap();
    assert_eq!(b.eval_decimal(16).unwrap(), "-5.318704469415522e1769");
}

// ═══════════════════════════════════════════════════════════════════════════
// Round 2: exact folds of eval that did not terminate or cancelled
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn beta_of_huge_integers_is_a_small_binomial() {
    // B(a, b) multiplied out (a−1)!(b−1)!/(a+b−1)!: these never returned.
    // mpmath (dps 40): beta(6, 10**7) = 1.199998200001679998740001e-40,
    // beta(10**12, 3) = 1.999999999994000000000014e-36,
    // beta(8, 87178291200) = 1.510649529514917539788258e-84,
    // beta(10**8, 17) = 2.092276143302666672974289e-123.
    let ctx = Context::new();
    let start = Instant::now();
    for (f, want) in [
        ("beta(6, 10^7)", "1.199998200001679998740001e-40"),
        ("beta(10^12, 3)", "1.999999999994000000000014e-36"),
        ("beta(8, factorial(14))", "1.510649529514917539788258e-84"),
        ("beta(10^8, 17)", "2.092276143302666672974289e-123"),
    ] {
        let e = ctx.parse(f).unwrap();
        assert!(!e.eval().to_string().contains("B("), "{f} stayed symbolic");
        assert_eq!(e.eval_decimal(25).unwrap(), want, "{f}");
    }
    assert_eq!(ctx.parse("beta(3, 4)").unwrap().eval().to_string(), "1/60");
    assert!(
        start.elapsed() < Duration::from_secs(20),
        "took {:?}",
        start.elapsed()
    );
}

#[test]
fn orthogonal_polynomials_of_high_degree() {
    // Each recurrence step was expanded and re-canonicalised: legendre(300, x)
    // took 6 s in eval (release), legendre(1000, x) and
    // jacobi(1000, 1/3, 1/5, x) more than a minute, and
    // chebyshev_t(1000, 1/10^20) 11 s before being refused.  (Degrees kept
    // moderate here: the test binary is a debug build.)
    // mpmath (dps 40): legendre(1000, 1/3) = 0.01961873093750094969323576,
    // jacobi(200, 1/3, 1/5, 1/7) = -0.05650541080748525966745,
    // gegenbauer(200, 1/3, 1/7) = -0.015689837250617792596,
    // laguerre(200, 1/3, 1/7) = -0.8757101127726804008133.
    let ctx = Context::new();
    let start = Instant::now();
    for (f, want) in [
        ("legendre(1000, 1/3)", "0.01961873093750094969323576"),
        ("jacobi(200, 1/3, 1/5, 1/7)", "-0.05650541080748525966745"),
        ("gegenbauer(200, 1/3, 1/7)", "-0.015689837250617792596"),
        ("assoc_laguerre(200, 1/3, 1/7)", "-0.8757101127726804008133"),
    ] {
        let digits = want.trim_start_matches(['-', '0', '.']).len() as u32;
        assert_eq!(
            ctx.parse(f).unwrap().eval_decimal(digits).unwrap(),
            want,
            "{f}"
        );
    }
    for f in [
        "legendre(400, x)",
        "hermite(400, x)",
        "assoc_legendre(400, 3, x)",
    ] {
        let e = ctx.parse(f).unwrap().eval();
        assert!(
            e.to_string()
                .starts_with(|c: char| c.is_ascii_digit() || c == '-'),
            "{f}"
        );
    }
    // Refused within the guard instead of built.
    let e = ctx.parse("chebyshev_t(1000, 1/10^20)").unwrap().eval();
    assert!(e.to_string().starts_with("chebyshev_t"), "{e}");
    assert!(
        start.elapsed() < Duration::from_secs(120),
        "took {:?}",
        start.elapsed()
    );
}

#[test]
fn explicit_coefficients_agree_with_the_symbolic_parameter_path() {
    // Numeric parameters take the explicit coefficients (DLMF §18.5), symbolic
    // ones the recurrences: the two agree once the parameters are substituted.
    // SymPy 1.14: expand(jacobi(2, 1/3, 1/5, x)) = 901*x**2/450 + 53*x/450 - 127/225,
    // expand(gegenbauer(3, 1/3, x)) = 112*x**3/81 - 8*x/9,
    // simplify(assoc_legendre(3, -1, x)) = sqrt(1 - x**2)*(5*x**2 - 1)/8.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let (ra, rb) = (ctx.rational(1, 3), ctx.rational(1, 5));
    for n in 0..=8 {
        for (sym, num) in [
            (
                format!("jacobi({n}, a, b, x)"),
                format!("jacobi({n}, 1/3, 1/5, x)"),
            ),
            (
                format!("gegenbauer({n}, a, x)"),
                format!("gegenbauer({n}, 1/3, x)"),
            ),
            (
                format!("assoc_laguerre({n}, a, x)"),
                format!("assoc_laguerre({n}, 1/3, x)"),
            ),
        ] {
            let s = ctx
                .parse(&sym)
                .unwrap()
                .eval()
                .subs(&a, &ra)
                .subs(&b, &rb)
                .expand();
            let v = ctx.parse(&num).unwrap().eval();
            assert!(
                (&s - &v).expand().is_zero_structural(),
                "{sym} ≠ {num}: {s} vs {v}"
            );
        }
    }
    let e = |f: &str| ctx.parse(f).unwrap().eval();
    let d = &e("jacobi(2, 1/3, 1/5, x)") - &ctx.parse("901*x^2/450 + 53*x/450 - 127/225").unwrap();
    assert!(d.expand().is_zero_structural());
    assert_eq!(e("gegenbauer(3, 1/3, x)").to_string(), "112/81*x^3 - 8/9*x");
    let p = e("assoc_legendre(3, -1, x)");
    let q = ctx.parse("sqrt(1 - x^2)*(5*x^2 - 1)/8").unwrap();
    assert!((&p - &q).expand().is_zero_structural(), "{p}");
    let _ = x;
}

#[test]
fn polygamma_closed_forms_do_not_cancel_beyond_evalf() {
    // ψ⁽ⁿ⁾(m + 1/2) as n!·(2^{n+1} − 1)ζ(n+1) minus the shift cancels
    // (n+1)·log₂(2m+1) bits: polygamma(3000, 3/2) printed 0.  The form is
    // now kept only when it cancels at most 200 bits (and n! is within the
    // digit guard); beyond, evalf evaluates the node.
    // (polygamma(1000, 3/2) printed 0 in the same way; 1000 keeps the debug
    // test binary fast.)
    // mpmath (dps 40): psi(1000, 3/2) = -2.174172045054930885998e+2391,
    // psi(1000, 1/3) = -1.595953364036279500722e+3045,
    // psi(1000, 7/2) = -9.829509064514376661302e+2022,
    // psi(125, 3/2) = 1.22257980252164383120009e+187 (closed form, 200 bits),
    // psi(126, 3/2) = -1.026967034118180818208075e+189 (left symbolic).
    let ctx = Context::new();
    for (f, want) in [
        ("polygamma(1000, 3/2)", "-2.174172045054930885998e2391"),
        ("polygamma(1000, 1/3)", "-1.595953364036279500722e3045"),
        ("polygamma(1000, 7/2)", "-9.829509064514376661302e2022"),
        ("polygamma(125, 3/2)", "1.22257980252164383120009e187"),
        ("polygamma(126, 3/2)", "-1.026967034118180818208075e189"),
    ] {
        let digits = want
            .split('e')
            .next()
            .unwrap()
            .trim_start_matches('-')
            .len() as u32
            - 1;
        assert_eq!(
            ctx.parse(f).unwrap().eval_decimal(digits).unwrap(),
            want,
            "{f}"
        );
    }
    assert!(
        !ctx.parse("polygamma(125, 3/2)")
            .unwrap()
            .eval()
            .to_string()
            .contains("polygamma")
    );
    assert!(
        ctx.parse("polygamma(126, 3/2)")
            .unwrap()
            .eval()
            .to_string()
            .contains("polygamma")
    );
    // n! beyond the digit guard: polygamma(10^5, 1/2) built 10^5! (it hung).
    let start = Instant::now();
    let e = ctx.parse("polygamma(10^5, 1/2)").unwrap().eval();
    assert!(e.to_string().contains("polygamma"), "{e}");
    assert!(start.elapsed() < Duration::from_secs(5));
}

#[test]
fn a_pole_inside_the_summation_range() {
    // Σ_{k=1}^{n} 1/(3 − k) came out −H(n − 3) + H(−3), which evaluates
    // nowhere and is not the sum for n ≥ 3 (the term at k = 3 is undefined);
    // Σ_{k=1}^{∞} 1/(3 − k) "diverged to −∞"; Σ_{k=1}^{n} 1/((k−3)(k−4))
    // was −1/(n − 3) − 1/3.  A concrete range through the pole is zoo, a
    // symbolic or infinite one stays unevaluated (SymPy 1.14:
    // summation(1/(3 - k), (k, 1, 5)) = zoo, (k, 1, n) and (k, 1, oo)
    // unevaluated; summation(1/(3 - k), (k, 4, 6)) = -11/6).
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let t = ctx.parse("1/(3 - k)").unwrap();
    assert!(t.try_summation(&k, &ctx.int(1), &n).is_err());
    assert!(t.try_summation(&k, &ctx.int(1), &ctx.infinity()).is_err());
    assert_eq!(t.summation(&k, &ctx.int(1), &ctx.int(5)).to_string(), "zoo");
    assert_eq!(t.summation(&k, &ctx.int(1), &ctx.int(2)).to_string(), "3/2");
    let s = t.summation(&k, &ctx.int(4), &n);
    assert_eq!(s.subs_i64(&n, 6).eval().to_string(), "-11/6", "{s}");
    let t2 = ctx.parse("1/((k - 3)*(k - 4))").unwrap();
    assert!(t2.try_summation(&k, &ctx.int(1), &n).is_err());
    let t3 = ctx.parse("1/k").unwrap();
    assert!(t3.try_summation(&k, &ctx.int(0), &n).is_err());
    assert_eq!(t3.summation(&k, &ctx.int(1), &n).to_string(), "harmonic(n)");
}
