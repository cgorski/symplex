//! After 0.29 — the `series` contract (real asymptotic expansion, not a
//! Laurent series), and limits, series and ODEs that were refused although
//! a value exists: special functions at their poles, logarithmic
//! singularities and at infinity (Gruntz needs the function's expansion
//! there), linear ODEs with symbolic constant coefficients, and
//! `check_ode_solution` false negatives.
//!
//! Each test says what was wrong before; every reference value cites the
//! oracle call that produced it (SymPy 1.14; mpmath 1.3.0 at `mp.dps = 60`,
//! scripts `oracle_tests.py`, `oracle2.py` of the hunt).  Where SymPy itself
//! returns the limit unevaluated, the value is mpmath's.

use symplex::prelude::*;

fn limit(ctx: &Context, f: &str, point: &str, dir: Direction) -> String {
    let x = ctx.symbol("x");
    let r = ctx
        .parse(f)
        .unwrap()
        .try_limit_dir(&x, &ctx.parse(point).unwrap(), dir);
    match r {
        Ok(v) => v.to_string(),
        Err(e) => format!("Err({e})"),
    }
}

/// `|z|` of a certified value to 30 digits (`0` for an exact zero, which
/// the evaluator reports as a precision failure).
fn magnitude(e: &Ex) -> f64 {
    match e.eval_decimal(30) {
        Ok(s) => {
            let t = s.replace(' ', "");
            if let Some(body) = t.strip_suffix("*i") {
                let k = body
                    .char_indices()
                    .skip(1)
                    .filter(|&(i, c)| (c == '+' || c == '-') && !body[..i].ends_with('e'))
                    .map(|(i, _)| i)
                    .last();
                let (re, im) = match k {
                    Some(k) => (
                        body[..k].parse::<f64>().unwrap(),
                        body[k..].parse::<f64>().unwrap(),
                    ),
                    None => (0.0, body.parse::<f64>().unwrap_or(1.0)),
                };
                re.hypot(im)
            } else {
                t.parse::<f64>().unwrap().abs()
            }
        }
        Err(err) => {
            let m = err.to_string();
            assert!(m.contains("recision") || m.contains("zero"), "{e}: {m}");
            0.0
        }
    }
}

/// `|got − expected|` at `x = at` (a rational given as text), to 30 digits,
/// is below `1e−25`.
fn assert_same_at(ctx: &Context, got: &Ex, expected: &str, at: &str) {
    let x = ctx.symbol("x");
    assert!(!got.has_unevaluated(), "refused: {got}");
    let e = ctx.parse(expected).unwrap();
    let pt = ctx.parse(at).unwrap();
    let d = (got - &e).subs(&x, &pt);
    let v = magnitude(&d);
    assert!(v < 1e-25, "{got} vs {expected} at x = {at}: {v}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Limits: special functions at their singular points
// ═══════════════════════════════════════════════════════════════════════════

/// Were refused: Gruntz had no expansion of `Γ`, `ψ`, `ψ′`, `ζ` at their
/// poles (`Γ(ω) − 1/ω` cancels, and differentiating `ω·Γ(ω)` at 0 hits the
/// pole).
#[test]
fn poles_minus_principal_parts() {
    let ctx = Context::new();
    // SymPy: limit(gamma(x) - 1/x, x, 0) -> -EulerGamma
    assert_eq!(
        limit(&ctx, "gamma(x) - 1/x", "0", Direction::Both),
        "-EulerGamma"
    );
    // mpmath: digamma(1e-20) + 1e20 = -0.5772156649… (SymPy: unevaluated)
    assert_eq!(
        limit(&ctx, "digamma(x) + 1/x", "0", Direction::Both),
        "-EulerGamma"
    );
    // mpmath: psi(1, 1e-20) - 1e40 = 1.6449340668… = pi^2/6 (SymPy: unevaluated)
    assert_eq!(
        limit(&ctx, "polygamma(1, x) - 1/x^2", "0", Direction::Both),
        "1/6*pi^2"
    );
    // mpmath: zeta(1 + 1e-12) - 1e12 = 0.5772156649016… (SymPy: unevaluated)
    assert_eq!(
        limit(&ctx, "zeta(x) - 1/(x - 1)", "1", Direction::Both),
        "EulerGamma"
    );
    // SymPy: limit(zeta(x)*(x - 1), x, 1) is unevaluated; the residue of ζ at 1 is 1
    assert_eq!(limit(&ctx, "zeta(x)*(x - 1)", "1", Direction::Both), "1");
    // SymPy: limit(gamma(x)*x, x, 0) -> 1 (worked before; kept)
    assert_eq!(limit(&ctx, "gamma(x)*x", "0", Direction::Both), "1");
}

/// Were refused: no expansion of `Ei`, `li`, `Ci`, `Chi`, `Shi`, `K₀`, `Y₀`
/// at the zeros of their arguments (logarithmic singularities need
/// `ln ω` as a coefficient, as in SymPy's `logx`).
#[test]
fn logarithmic_singularities_minus_their_logarithm() {
    let ctx = Context::new();
    // SymPy: limit(Ei(x) - log(x), x, 0, '+') -> EulerGamma
    assert_eq!(
        limit(&ctx, "Ei(x) - ln(x)", "0", Direction::Right),
        "EulerGamma"
    );
    // SymPy: limit(Ei(x) - log(Abs(x)), x, 0) -> EulerGamma
    assert_eq!(
        limit(&ctx, "Ei(x) - ln(abs(x))", "0", Direction::Both),
        "EulerGamma"
    );
    // SymPy: limit(li(x) - log(Abs(log(x))), x, 1) -> EulerGamma
    assert_eq!(
        limit(&ctx, "li(x) - ln(abs(ln(x)))", "1", Direction::Both),
        "EulerGamma"
    );
    // SymPy: limit(Ci(x) - log(x), x, 0, '+') -> EulerGamma
    assert_eq!(
        limit(&ctx, "Ci(x) - ln(x)", "0", Direction::Right),
        "EulerGamma"
    );
    // SymPy: limit(Chi(x) - log(x), x, 0, '+') -> EulerGamma
    assert_eq!(
        limit(&ctx, "Chi(x) - ln(x)", "0", Direction::Right),
        "EulerGamma"
    );
    // SymPy: limit(Shi(x)/x, x, 0) -> 1
    assert_eq!(limit(&ctx, "Shi(x)/x", "0", Direction::Both), "1");
    // SymPy: limit(besselk(0, x) + log(x/2), x, 0, '+') -> -EulerGamma
    // (was also `-EulerGamma + ln(1/2) + ln(2)` once expanded: ln(p/q) is
    // now split so that the constants cancel)
    assert_eq!(
        limit(&ctx, "besselk(0, x) + ln(x/2)", "0", Direction::Right),
        "-EulerGamma"
    );
    // SymPy: limit(bessely(0, x) - 2/pi*log(x/2), x, 0, '+') -> 2*EulerGamma/pi
    assert_eq!(
        limit(&ctx, "bessely(0, x) - 2/pi*ln(x/2)", "0", Direction::Right),
        "2*EulerGamma/pi"
    );
    // SymPy: limit(loggamma(x) + log(x), x, 0, '+') -> 0
    assert_eq!(
        limit(&ctx, "loggamma(x) + ln(x)", "0", Direction::Right),
        "0"
    );
}

/// `W(x)/x` was refused (differentiating `W` at 0 gives `0/0`); `(sin x/x)^(1/x²)`
/// ran out of recursion depth after Gruntz split the power into two
/// exponentials of the same class.
#[test]
fn lambert_w_and_one_to_the_infinity() {
    let ctx = Context::new();
    // SymPy: limit(LambertW(x)/x, x, 0) -> 1
    assert_eq!(limit(&ctx, "LambertW(x)/x", "0", Direction::Both), "1");
    // SymPy: limit((sin(x)/x)**(1/x**2), x, 0) -> exp(-1/6)
    assert_eq!(
        limit(&ctx, "(sin(x)/x)^(1/x^2)", "0", Direction::Both),
        "exp(-1/6)"
    );
    // SymPy: limit(cos(x)**(1/x**2), x, 0) -> exp(-1/2)
    assert_eq!(
        limit(&ctx, "cos(x)^(1/x^2)", "0", Direction::Both),
        "exp(-1/2)"
    );
}

/// Were refused: `erfc` of a divergent argument counted as "unbounded", `Ei`,
/// `li` had no asymptotic series, `ln Γ`, `ψ` no Stirling series.  They are
/// now rewritten with their exponential factor explicit (SymPy's
/// "tractable" rewrite) and expanded asymptotically.
#[test]
fn asymptotics_at_infinity() {
    let ctx = Context::new();
    let oo = "oo";
    let b = Direction::Both;
    // SymPy: limit(erfc(x)*x*exp(x**2), x, oo) -> 1/sqrt(pi)
    assert_eq!(limit(&ctx, "erfc(x)*x*exp(x^2)", oo, b), "1/sqrt(pi)");
    // SymPy: limit(x*(1 - erf(x))*exp(x**2), x, oo) -> 1/sqrt(pi)
    assert_eq!(limit(&ctx, "x*(1 - erf(x))*exp(x^2)", oo, b), "1/sqrt(pi)");
    // SymPy: limit(Ei(x)*x*exp(-x), x, oo) -> 1
    assert_eq!(limit(&ctx, "Ei(x)*x*exp(-x)", oo, b), "1");
    // SymPy: limit(Ei(-x)*x*exp(x), x, oo) -> -1
    assert_eq!(limit(&ctx, "Ei(-x)*x*exp(x)", oo, b), "-1");
    // mpmath: Ei(1e8)*1e8*exp(-1e8) = 1 + 1e-8 + 2e-16 + …: the next term is 1/x
    assert_eq!(limit(&ctx, "x*(Ei(x)*x*exp(-x) - 1)", oo, b), "1");
    // SymPy: limit(li(x)*log(x)/x, x, oo) -> 1
    assert_eq!(limit(&ctx, "li(x)*ln(x)/x", oo, b), "1");
    // SymPy: limit(digamma(x) - log(x), x, oo) -> 0
    assert_eq!(limit(&ctx, "digamma(x) - ln(x)", oo, b), "0");
    // SymPy: limit(loggamma(x) - (x - 1/2)*log(x) + x, x, oo) -> log(2)/2 + log(pi)/2
    assert_eq!(
        limit(&ctx, "loggamma(x) - (x - 1/2)*ln(x) + x", oo, b),
        "1/2*ln(2) + 1/2*ln(pi)"
    );
    // SymPy: limit(loggamma(x + 1) - loggamma(x) - log(x), x, oo) -> 0
    // (identically zero: every coefficient of its expansion vanishes)
    assert_eq!(
        limit(&ctx, "loggamma(x + 1) - loggamma(x) - ln(x)", oo, b),
        "0"
    );
}

/// Wrong values before: `Γ`, `Ei` and the like were classed with `x` by the
/// MRV set, and when `eˣ` dominated they were left in the leading
/// coefficient as if of lower order.  `Γ` at `∞` is now `exp(ln Γ)`, `Ei`
/// is `eˣ` times its asymptotic series, and a function of unknown growth is
/// refused instead.
#[test]
fn functions_of_exponential_growth_are_not_taken_for_powers() {
    let ctx = Context::new();
    let (oo, b) = ("oo", Direction::Both);
    // was 0.  SymPy: limit(gamma(x)/exp(x), x, oo) -> oo
    assert_eq!(limit(&ctx, "gamma(x)/exp(x)", oo, b), "oo");
    // was 0.  SymPy: limit(factorial(x)/(x**x*exp(-x)*sqrt(2*pi*x)), x, oo) -> 1
    assert_eq!(limit(&ctx, "x!/(x^x*exp(-x)*sqrt(2*pi*x))", oo, b), "1");
    // was 0.  SymPy: limit(factorial(x)*exp(x)/x**x, x, oo) -> oo
    assert_eq!(limit(&ctx, "x!*exp(x)/x^x", oo, b), "oo");
    // was 0 (Ei(2x)·2x·e^{−2x}); mpmath: ei(2e8)*2e8*exp(-2e8) = 1.000000005
    assert_eq!(limit(&ctx, "Ei(2*x)*2*x*exp(-2*x)", oo, b), "1");
    // was -oo; mpmath: Ei(x)·x·e⁻ˣ = 1 + 1/x + 2/x² + …, so 1/2 for x → 2x
    assert_eq!(limit(&ctx, "x*(Ei(2*x)*2*x*exp(-2*x) - 1)", oo, b), "1/2");
    // were refused (Gruntz looped on e^{ln Γ(x+1) − ln Γ(x)}); SymPy: 1, 1, 0, oo
    assert_eq!(limit(&ctx, "gamma(x + 1)/(x*gamma(x))", oo, b), "1");
    assert_eq!(limit(&ctx, "gamma(x + 1/2)/(gamma(x)*sqrt(x))", oo, b), "1");
    assert_eq!(limit(&ctx, "x!/x^x", oo, b), "0");
    assert_eq!(limit(&ctx, "x^x/x!", oo, b), "oo");
    // Were 0; the true values are 1/2, 1/2, 1/√(2π), 1/√π (mpmath at
    // x = 10⁶, 40 digits: chi(X)*X*exp(-X) = 0.5000005, shi likewise,
    // besseli(0,X)*exp(-X)*sqrt(X) = 0.39894233, erfi(X)*X*exp(-X**2) =
    // 0.56418958; SymPy 1.14 also answers 0 for the first, wrongly).  Their
    // growth is not known to the engine: refused, not wrong.
    for f in [
        "Chi(x)*x*exp(-x)",
        "Shi(x)*x*exp(-x)",
        "besseli(0, x)*exp(-x)*sqrt(x)",
        "erfi(x)*x*exp(-x^2)",
    ] {
        assert!(limit(&ctx, f, oo, b).starts_with("Err"), "{f}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Series at singular points and at infinity
// ═══════════════════════════════════════════════════════════════════════════

/// Were refused (formal `Series` nodes): `Γ` and `ψ` at their poles.
#[test]
fn series_of_gamma_and_digamma_at_poles() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // SymPy: series(gamma(x), x, 0, 4)
    let s = x.gamma().series(&x, &ctx.int(0), 4);
    assert_same_at(
        &ctx,
        &s,
        "1/x - EulerGamma + x*(EulerGamma^2/2 + pi^2/12) \
         + x^2*(-EulerGamma*pi^2/12 - zeta(3)/3 - EulerGamma^3/6) \
         + x^3*(EulerGamma^4/24 + EulerGamma^2*pi^2/24 + EulerGamma*zeta(3)/3 + pi^4/160)",
        "1/7",
    );
    // SymPy: series(gamma(x), x, -2, 3)
    let s = x.gamma().series(&x, &ctx.int(-2), 3);
    assert_same_at(
        &ctx,
        &s,
        "1/(2*(x + 2)) + 3/4 - EulerGamma/2 \
         + (x + 2)*(-3*EulerGamma/4 + EulerGamma^2/4 + pi^2/24 + 7/8) \
         + (x + 2)^2*(-7*EulerGamma/8 - EulerGamma*pi^2/24 - zeta(3)/6 - EulerGamma^3/12 \
         + 3*EulerGamma^2/8 + pi^2/16 + 15/16)",
        "-13/7",
    );
    // DLMF 5.7.6; mpmath: taylor(digamma(t) + 1/t, 0, 2) =
    // [-0.5772156649…, 1.6449340668…, -1.2020569031…] (SymPy: PoleError)
    let s = x.digamma().series(&x, &ctx.int(0), 3);
    assert_same_at(
        &ctx,
        &s,
        "-1/x - EulerGamma + pi^2/6*x - zeta(3)*x^2",
        "1/7",
    );
}

/// `ζ` at its pole: `1/(x − 1) + γ` is exact below order 1; beyond, the
/// coefficients are Stieltjes constants (`γ₁ = −0.0728158454…`, mpmath
/// `stieltjes(1)`), which have no representation, so higher orders stay
/// refused (SymPy: `series(zeta(x), x, 1, 3)` raises `PoleError`).
#[test]
fn series_of_zeta_at_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = x.zeta().series(&x, &ctx.int(1), 1);
    assert_same_at(&ctx, &s, "1/(x - 1) + EulerGamma", "8/7");
    assert!(x.zeta().series(&x, &ctx.int(1), 3).has_unevaluated());
}

/// Stirling's series: `ln Γ` and `ψ` at `∞` were refused; the expansion at
/// `±∞` is now log-extended (coefficients may contain `ln x`).
#[test]
fn stirling_series_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let oo = ctx.infinity();
    // SymPy: series(loggamma(x), x, oo, 4) ->
    //   -1/(360*x**3) + 1/(12*x) + log(2*pi)/2 + log(1/x)/2 + x*(-log(1/x) - 1)
    let s = x.log_gamma().series(&x, &oo, 4);
    assert_same_at(
        &ctx,
        &s,
        "-1/(360*x^3) + 1/(12*x) + ln(2*pi)/2 - ln(x)/2 + x*(ln(x) - 1)",
        "7",
    );
    // SymPy: series(digamma(x), x, oo, 5) ->
    //   1/(120*x**4) - 1/(12*x**2) - 1/(2*x) - log(1/x)
    let s = x.digamma().series_at_infinity(&x, 5);
    assert_same_at(&ctx, &s, "1/(120*x^4) - 1/(12*x^2) - 1/(2*x) + ln(x)", "7");
}

/// The contract: a real asymptotic expansion, two-sided unless a direction
/// is given.  `series_dir` is new; the two-sided results are unchanged.
#[test]
fn series_contract_essential_singularities_and_sides() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    // SymPy: series(exp(-1/x**2), x, 0, 4) -> O(x**4)
    let f = (-(x.powi(-2))).exp();
    assert_eq!(f.series(&x, &zero, 4).to_string(), "0");
    // exp(-1/x): O(x^4) from the right, unbounded from the left.
    // SymPy: series(exp(-1/x), x, 0, 4) -> O(x**4) (default dir='+');
    //        series(exp(-1/x), x, 0, 4, dir='-') -> exp(-1/x) (unexpanded)
    let g = (-(1 / &x)).exp();
    assert!(g.series(&x, &zero, 4).has_unevaluated());
    assert_eq!(
        g.series_dir(&x, &zero, 4, Direction::Right).to_string(),
        "0"
    );
    assert!(
        g.series_dir(&x, &zero, 4, Direction::Left)
            .has_unevaluated()
    );
    assert!(g.try_series_dir(&x, &zero, 4, Direction::Left).is_err());
    // |x| one-sided (SymPy: series(Abs(x), x, 0, 3) -> x; dir='-' -> -x)
    assert_eq!(x.abs().series_dir(&x, &zero, 3, Direction::Right), x);
    assert_eq!(x.abs().series_dir(&x, &zero, 3, Direction::Left), -&x);
    // SymPy: series(Ei(x), x, 0, 3) -> EulerGamma + log(x) + x + x**2/4
    let s = x.ei().series_dir(&x, &zero, 3, Direction::Right);
    assert_same_at(&ctx, &s, "EulerGamma + ln(x) + x + x^2/4", "1/7");
    assert!(x.ei().series(&x, &zero, 3).has_unevaluated());
    // SymPy: series(besselk(0, x), x, 0, 3) ->
    //   log(2) - EulerGamma - log(x) + x**2*(-log(x)/4 - EulerGamma/4 + log(2)/4 + 1/4)
    let k0 = ctx.parse("besselk(0, x)").unwrap();
    let s = k0.series_dir(&x, &zero, 3, Direction::Right);
    assert_same_at(
        &ctx,
        &s,
        "ln(2) - EulerGamma - ln(x) + x^2*(-ln(x)/4 - EulerGamma/4 + ln(2)/4 + 1/4)",
        "1/7",
    );
    // SymPy: series(Shi(x), x, 0, 6) -> x + x**3/18 + x**5/600
    let shi = ctx.parse("Shi(x)").unwrap();
    assert_eq!(
        shi.series(&x, &zero, 6).to_string(),
        "1/600*x^5 + 1/18*x^3 + x"
    );
}

/// `laurent_series_expr` took the real expansion of `series`: the "Laurent
/// series" of `exp(−1/x²)` at 0 was `0`.  An essential singularity is now
/// refused; poles still expand.
#[test]
fn laurent_series_refuses_an_essential_singularity() {
    let ctx = Context::new();
    ctx.with_arena_mut(|a| {
        let x = a.symbol("x");
        let zero = a.zero();
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one();
        let inv = a.div(one, x2);
        let arg = a.neg(inv);
        let f = a.exp(arg);
        assert!(a.laurent_series_expr(f, x, zero, 4).is_err());
        // 1/sin(x) = 1/x + x/6 + 7x³/360 + … (SymPy: series(1/sin(x), x, 0, 4))
        let s = a.sin(x);
        let g = a.div(one, s);
        let r = a.laurent_series_expr(g, x, zero, 4).unwrap();
        assert_eq!(a.display(r).to_string(), "7/360*x^3 + 1/x + 1/6*x");
    });
}

/// `ln(−1 + i·x)` lies on the cut of `ln` at `x = 0`: it tends to `iπ` from
/// `x > 0` and to `−iπ` from `x < 0`, so there is no two-sided expansion.
/// Before, the series was `iπ − i·x` on both sides.  (SymPy:
/// `series(log(-1 + I*x), x, 0, 2) -> I*pi - I*x`, its default being the
/// right-hand side; mpmath: `log(-1 - 1e-20j) = -3.14159…j`.)
#[test]
fn series_on_a_branch_cut_is_refused_two_sided() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("ln(-1 + I*x)").unwrap();
    assert!(f.series(&x, &ctx.int(0), 2).has_unevaluated());
    // on the real axis the cut is approached along itself: unchanged
    let g = ctx.parse("ln(-1 + x)").unwrap();
    assert_eq!(g.series(&x, &ctx.int(0), 2).to_string(), "-x + pi*I");
}

// ═══════════════════════════════════════════════════════════════════════════
// ODEs
// ═══════════════════════════════════════════════════════════════════════════

/// The ODE text with `D1Y`, `D2Y`, `D3Y` for `y′`, `y″`, `y‴` in `x`.
fn ode(ctx: &Context, text: &str) -> Ex {
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let d1 = y.formal_diff(&x);
    let d2 = d1.formal_diff(&x);
    let d3 = d2.formal_diff(&x);
    ctx.parse(text)
        .unwrap()
        .subs(&ctx.symbol("D3Y"), &d3)
        .subs(&ctx.symbol("D2Y"), &d2)
        .subs(&ctx.symbol("D1Y"), &d1)
}

/// The residual of `sol` in the ODE `text = 0`, with `a = 3/5` (or `-7/4`)
/// and the constants set to rational values, vanishes at two points.
fn assert_solves(ctx: &Context, text: &str, sol: &Ex) {
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    assert!(!sol.contains(&y) && !sol.has_unevaluated(), "{text}: {sol}");
    let s1 = sol.diff(&x);
    let s2 = s1.diff(&x);
    let s3 = s2.diff(&x);
    let resid = ctx
        .parse(text)
        .unwrap()
        .subs(&ctx.symbol("D3Y"), &s3)
        .subs(&ctx.symbol("D2Y"), &s2)
        .subs(&ctx.symbol("D1Y"), &s1)
        .subs(&y, sol);
    for a in [(3, 5), (-7, 4)] {
        for (p, q) in [(1, 3), (-5, 7)] {
            let r = resid
                .subs(&ctx.symbol("a"), &ctx.rational(a.0, a.1))
                .subs(&ctx.symbol("C1"), &ctx.rational(2, 3))
                .subs(&ctx.symbol("C2"), &ctx.rational(-1, 5))
                .subs(&ctx.symbol("C3"), &ctx.rational(7, 2))
                .subs(&x, &ctx.rational(p, q));
            let v = magnitude(&r);
            assert!(
                v < 1e-20,
                "{text}: {sol} residual {v} at a = {a:?}, x = {p}/{q}"
            );
        }
    }
}

/// Linear ODEs with a symbolic constant coefficient were refused
/// (`DSolve(…)`): the solvers took rational coefficients only.  The roots
/// of the characteristic polynomial are taken in radicals, principal
/// branches, with no case split on the sign of `a` (SymPy's choice).
#[test]
fn constant_coefficients_with_a_symbolic_parameter() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for (text, want) in [
        // SymPy: dsolve(y'' + a*y) -> C1*exp(-x*sqrt(-a)) + C2*exp(x*sqrt(-a))
        ("D2Y + a*y", "C1*exp(x*sqrt(-a)) + C2*exp(-x*sqrt(-a))"),
        // SymPy: C1*exp(x*(-a + sqrt(a**2 - 4))/2) + C2*exp(-x*(a + sqrt(a**2 - 4))/2)
        (
            "D2Y + a*D1Y + y",
            "C1*exp(x*(-1/2*a + 1/2*sqrt(a^2 - 4))) + C2*exp(x*(-1/2*a - 1/2*sqrt(a^2 - 4)))",
        ),
        // SymPy: (C1 + C2*x)*exp(-a*x)
        ("D2Y + 2*a*D1Y + a^2*y", "C2*x*exp(-a*x) + C1*exp(-a*x)"),
        // SymPy: C1 + C2*exp(-x*sqrt(-a)) + C3*exp(x*sqrt(-a))
        (
            "D3Y + a*D1Y",
            "C1 + C2*exp(x*sqrt(-a)) + C3*exp(-x*sqrt(-a))",
        ),
        // SymPy: dsolve(y'' + (a + 1)*y' + a*y) -> C1*exp(-a*x) + C2*exp(-x)
        ("D2Y + (a + 1)*D1Y + a*y", "C1*exp(-x) + C2*exp(-a*x)"),
        // SymPy: dsolve(y'' + a*y - x) -> C1*exp(-x*sqrt(-a)) + C2*exp(x*sqrt(-a)) + x/a
        (
            "D2Y + a*y - x",
            "x/a + C1*exp(x*sqrt(-a)) + C2*exp(-x*sqrt(-a))",
        ),
    ] {
        let e = ode(&ctx, text);
        let sol = e.solve_ode(&y, &x);
        assert_eq!(sol.to_string(), want, "{text}");
        assert!(e.check_ode_solution(&sol, &y, &x), "{text}: {sol}");
        assert_solves(&ctx, text, &sol);
    }
    // A forcing term the undetermined coefficients of the rational solver
    // would take: variation of parameters, generic branch (a ≠ −1).
    let text = "D2Y + a*y - exp(x)";
    let e = ode(&ctx, text);
    let sol = e.solve_ode(&y, &x);
    assert!(e.check_ode_solution(&sol, &y, &x), "{sol}");
    assert_solves(&ctx, text, &sol);
}

/// `check_ode_solution` rejected correct solutions: the Bernoulli family
/// needs `B^{−1/2} = B·B^{−3/2}` for the radical base `B`, the
/// variation-of-parameters solution of `y″ + y = tan x` needs `sin² + cos² =
/// 1` over a common denominator.  Wrong solutions stay rejected.
#[test]
fn check_ode_solution_radicals_and_trig() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // SymPy: checkodesol(y' + y - y**3, 1/sqrt((C1 + exp(-2*x))*exp(2*x))) -> (True, 0)
    let bern = ode(&ctx, "D1Y + y - y^3");
    let good = ctx.parse("1/sqrt((C1 + exp(-2*x))*exp(2*x))").unwrap();
    assert!(bern.check_ode_solution(&good, &y, &x));
    for bad in [
        "2/sqrt((C1 + exp(-2*x))*exp(2*x))",
        "1/sqrt((C1 + exp(-2*x))*exp(x))",
        "1/sqrt((C1 + exp(-x))*exp(2*x))",
        "(C1 + exp(-2*x))*exp(2*x)",
    ] {
        let s = ctx.parse(bad).unwrap();
        assert!(!bern.check_ode_solution(&s, &y, &x), "accepted {bad}");
    }
    // The IVP y(0) = 0, y'(0) = 1 of y'' + y = tan x (variation of
    // parameters); substituted by hand: y'' + y - tan x = 0.
    let tan_ode = ode(&ctx, "D2Y + y - tan(x)");
    let good = ctx
        .parse("-cos(x)*ln(abs(1/cos(x) + tan(x))) + 2*sin(x)")
        .unwrap();
    assert!(tan_ode.check_ode_solution(&good, &y, &x));
    let ivp = tan_ode
        .solve_ode_ivp(
            &y,
            &x,
            &[
                InitialCondition {
                    order: 0,
                    x: ctx.int(0),
                    value: ctx.int(0),
                },
                InitialCondition {
                    order: 1,
                    x: ctx.int(0),
                    value: ctx.int(1),
                },
            ],
        )
        .unwrap();
    assert!(tan_ode.check_ode_solution(&ivp, &y, &x), "{ivp}");
    for bad in [
        "cos(x)*ln(abs(1/cos(x) + tan(x))) + 2*sin(x)",
        "-cos(x)*ln(abs(1/cos(x) + tan(x))) + 2*sin(x) + x",
        "-sin(x)*ln(abs(1/cos(x) + tan(x))) + 2*cos(x)",
    ] {
        let s = ctx.parse(bad).unwrap();
        assert!(!tan_ode.check_ode_solution(&s, &y, &x), "accepted {bad}");
    }
    // The branches of y' = y^3 (just added) are unaffected.
    let cubic = ode(&ctx, "D1Y - y^3");
    let all = cubic.solve_ode_all(&y, &x).unwrap();
    assert_eq!(all.len(), 2);
    assert!(all.iter().all(|s| cubic.check_ode_solution(s, &y, &x)));
}
