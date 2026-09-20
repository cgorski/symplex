//! symplex 0.12 — generalised incomplete beta `betainc(a, b, x1, x2)` and
//! `betainc_regularized(a, b, x1, x2)` (SymPy's 4-argument form; the `Ex`
//! method lives on the upper limit: `x2.betainc(&a, &b, &x1)`).
//!
//! Reference values cite SymPy 1.14 / mpmath 1.3 (`symplex/.venv/bin/python`,
//! `mp.dps = 60`).

use symplex::prelude::*;

/// Round trip through the parser.
fn roundtrip(ctx: &Context, e: &Ex) {
    let s = format!("{e}");
    let back = ctx
        .parse(&s)
        .unwrap_or_else(|err| panic!("parse({s}) failed: {err}"));
    assert_eq!(back, *e, "parse round trip of `{s}`");
}

/// `e.eval_decimal(digits)` starts with `prefix` (the first `digits`
/// significant digits of the mpmath reference).
fn assert_digits(e: &Ex, digits: u32, prefix: &str) {
    let s = e
        .eval_decimal(digits)
        .unwrap_or_else(|err| panic!("eval_decimal({e}, {digits}) failed: {err}"));
    assert!(
        s.starts_with(prefix),
        "{e} at {digits} digits: {s} !~ {prefix}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact folds
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exact_polynomial_folds_for_integer_parameters() {
    let ctx = Context::new();
    let (x, x1) = (ctx.symbol("x"), ctx.symbol("x1"));
    let zero = ctx.int(0);

    // sympy: cdf(Beta(2, 3))(x) = 3*x**4 - 8*x**3 + 6*x**2
    let cdf = x
        .betainc_regularized(&ctx.int(2), &ctx.int(3), &zero)
        .eval();
    assert_eq!(format!("{cdf}"), "3*x^4 - 8*x^3 + 6*x^2");
    // sympy: cdf(Beta(3, 4))(x) = -10*x**6 + 36*x**5 - 45*x**4 + 20*x**3
    let cdf = x
        .betainc_regularized(&ctx.int(3), &ctx.int(4), &zero)
        .eval();
    assert_eq!(format!("{cdf}"), "-10*x^6 + 36*x^5 - 45*x^4 + 20*x^3");
    // sympy: integrate((1-t)**4, (t, 0, x))/beta(1, 5) = x**5 - 5*x**4 + 10*x**3 - 10*x**2 + 5*x
    let cdf = x
        .betainc_regularized(&ctx.int(1), &ctx.int(5), &zero)
        .eval();
    assert_eq!(format!("{cdf}"), "x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x");
    // Uniform: B_{(0, x)}(1, 1) = x
    assert_eq!(x.betainc(&ctx.int(1), &ctx.int(1), &zero).eval(), x);

    // Unregularised, both limits symbolic.
    // sympy: integrate(t*(1-t)**2, (t, x1, x)) = x**4/4 - 2*x**3/3 + x**2/2 - x1**4/4 + 2*x1**3/3 - x1**2/2
    let b = x.betainc(&ctx.int(2), &ctx.int(3), &x1).eval();
    let expected = (x.powi(4) / 4 - 2 * x.powi(3) / 3 + x.powi(2) / 2 - x1.powi(4) / 4
        + 2 * x1.powi(3) / 3
        - x1.powi(2) / 2)
        .expand();
    assert_eq!(b, expected, "{b}");

    // Numeric limits fold to exact rationals.
    // sympy: integrate(t*(1-t)**2, (t, 0, 2/5)) = 82/1875; /beta(2,3) = 328/625
    let two_fifths = ctx.rational(2, 5);
    assert_eq!(
        format!(
            "{}",
            two_fifths.betainc(&ctx.int(2), &ctx.int(3), &zero).eval()
        ),
        "82/1875"
    );
    assert_eq!(
        format!(
            "{}",
            two_fifths
                .betainc_regularized(&ctx.int(2), &ctx.int(3), &zero)
                .eval()
        ),
        "328/625"
    );
    // sympy: integrate(t*(1-t)**2, (t, 1/5, 2/5))/beta(2, 3) = 43/125
    assert_eq!(
        format!(
            "{}",
            two_fifths
                .betainc_regularized(&ctx.int(2), &ctx.int(3), &ctx.rational(1, 5))
                .eval()
        ),
        "43/125"
    );

    // Non-integer or symbolic parameters stay unevaluated.
    let half = ctx.rational(1, 2);
    let g = x.betainc(&half, &ctx.int(3), &zero);
    assert_eq!(g.eval(), g);
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let g = x.betainc_regularized(&a, &b, &zero);
    assert_eq!(g.eval(), g);
}

#[test]
fn exact_special_values() {
    let ctx = Context::new();
    let (a, b, x) = (ctx.symbol("a"), ctx.symbol("b"), ctx.symbol("x"));
    let (zero, one) = (ctx.int(0), ctx.int(1));

    // Coincident limits.
    assert_eq!(x.betainc(&a, &b, &x).eval(), zero);
    assert_eq!(x.betainc_regularized(&a, &b, &x).eval(), zero);
    // Full interval: B(a, b) resp. 1 (sympy: betainc(2, 3, 0, 1) → beta(2, 3) = 1/12).
    assert_eq!(one.betainc(&a, &b, &zero).eval(), a.beta(&b));
    assert_eq!(one.betainc_regularized(&a, &b, &zero).eval(), one);
    assert_eq!(
        format!("{}", one.betainc(&ctx.int(2), &ctx.int(3), &zero).eval()),
        "1/12"
    );
    assert_eq!(
        one.betainc_regularized(&ctx.rational(1, 2), &ctx.rational(7, 3), &zero)
            .eval(),
        one
    );
    // The integral diverges for a ≤ 0: no fold for parameters known to be non-positive.
    let g = one.betainc(&ctx.int(-1), &b, &zero);
    assert_eq!(g.eval(), g);
    let g = one.betainc_regularized(&a, &zero, &zero);
    assert_eq!(g.eval(), g);
}

// ═══════════════════════════════════════════════════════════════════════════
// Derivatives
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn derivatives() {
    let ctx = Context::new();
    let (a, b, x, x1) = (
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("x"),
        ctx.symbol("x1"),
    );
    let f = x.betainc(&a, &b, &x1);
    // sympy: diff(betainc(a, b, x1, x2), x2) = x2**(a - 1)*(1 - x2)**(b - 1)
    let d_upper = x.pow(&(&a - 1)) * (1 - &x).pow(&(&b - 1));
    assert_eq!(f.diff(&x), d_upper);
    // sympy: diff(betainc(a, b, x1, x2), x1) = -x1**(a - 1)*(1 - x1)**(b - 1)
    let d_lower = -(x1.pow(&(&a - 1)) * (1 - &x1).pow(&(&b - 1)));
    assert_eq!(f.diff(&x1), d_lower);
    // Regularised: divided by B(a, b).
    let g = x.betainc_regularized(&a, &b, &x1);
    assert_eq!(g.diff(&x), &d_upper / a.beta(&b));
    assert_eq!(g.diff(&x1), &d_lower / a.beta(&b));
    // Parameter derivatives stay formal (sympy: Derivative(betainc(a, b, x1, x2), a)).
    assert_eq!(
        format!("{}", f.diff(&a)),
        "Derivative(betainc(a, b, x1, x), a)"
    );
    // Chain rule through the upper limit.
    // sympy: diff(betainc(a, b, x1, x**2), x) = 2*x*(1 - x**2)**(b - 1)*(x**2)**(a - 1)
    let h = x.powi(2).betainc(&a, &b, &x1).diff(&x);
    let expected = 2 * &x * x.powi(2).pow(&(&a - 1)) * (1 - x.powi(2)).pow(&(&b - 1));
    assert_eq!(h, expected, "{h}");
    // Nothing depends on y.
    let y = ctx.symbol("y");
    assert_eq!(f.diff(&y), ctx.int(0));
    // The derivative of the integer-parameter polynomial agrees with the
    // integrand (`B(2, 3) = 1/12` folds under `eval`).
    let cdf = x.betainc_regularized(&ctx.int(2), &ctx.int(3), &ctx.int(0));
    let lhs = cdf.eval().diff(&x);
    let rhs = cdf.diff(&x).eval().expand();
    assert_eq!(lhs, rhs, "{lhs} vs {rhs}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Numerical evaluation (arbitrary precision)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_matches_mpmath_at_50_digits() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    // mpmath: betainc(0.5, 0.5, 0, 0.3, regularized=True)
    //   = 0.3690101195655453827554305587787365146547243053879800607
    assert_digits(
        &p("betainc_regularized(1/2, 1/2, 0, 3/10)"),
        50,
        "0.36901011956554538275543055877873651465472430538798",
    );
    // mpmath: betainc(2.5, 3.7, 0, 0.6)
    //   = 0.0275434080321464513773697250514224289814188538051822594
    assert_digits(
        &p("betainc(5/2, 37/10, 0, 3/5)"),
        50,
        "0.027543408032146451377369725051422428981418853805182",
    );
    // small parameters — mpmath: betainc(0.1, 0.7, 0, 0.5, regularized=True)
    //   = 0.8944582691314096327990111656272699889412473130427560931
    assert_digits(
        &p("betainc_regularized(1/10, 7/10, 0, 1/2)"),
        50,
        "0.89445826913140963279901116562726998894124731304276",
    );
    // mpmath: betainc(0.1, 0.1, 0, 0.02) = 6.773610773161804556150410462044937748352331021724172251
    assert_digits(
        &p("betainc(1/10, 1/10, 0, 1/50)"),
        50,
        "6.7736107731618045561504104620449377483523310217242",
    );
    // large parameters — mpmath: betainc(50, 50, 0, 0.55, regularized=True)
    //   = 0.8413478010629012053023479902984694168086808006223948241
    assert_digits(
        &p("betainc_regularized(50, 50, 0, 11/20)"),
        50,
        "0.84134780106290120530234799029846941680868080062239",
    );
    // x near 1, large a — mpmath: betainc(50, 2.5, 0, 0.99, regularized=True)
    //   = 0.9609357502320382278146963862437407253756792783887690117
    assert_digits(
        &p("betainc_regularized(50, 5/2, 0, 99/100)"),
        50,
        "0.96093575023203822781469638624374072537567927838877",
    );
    // x near 1, small b — mpmath: betainc(20, 0.3, 0, 0.99, regularized=True)
    //   = 0.344606340457720534316840845696852025011645137639651739
    assert_digits(
        &p("betainc_regularized(20, 3/10, 0, 99/100)"),
        50,
        "0.34460634045772053431684084569685202501164513763965",
    );
    // mpmath: betainc(20, 0.3, 0, 0.999, regularized=True)
    //   = 0.658726499128895397529985681668029303457818807915183066
    assert_digits(
        &p("betainc_regularized(20, 3/10, 0, 999/1000)"),
        50,
        "0.65872649912889539752998568166802930345781880791518",
    );
    // both limits interior, half-integer b — mpmath: betainc(7, 11.5, 0.3, 0.35, regularized=True)
    //   = 0.1650324488357666740419643704093952374696670286770491062
    assert_digits(
        &p("betainc_regularized(7, 23/2, 3/10, 7/20)"),
        50,
        "0.16503244883576667404196437040939523746966702867705",
    );
    // both limits interior — mpmath: betainc(1.5, 0.5, 0.2, 0.9)
    //   = 0.8853981633974483096156608458198757210492923498437764552
    assert_digits(
        &p("betainc(3/2, 1/2, 1/5, 9/10)"),
        50,
        "0.88539816339744830961566084581987572104929234984378",
    );
    // upper limit 1 — mpmath: betainc(2.5, 3.7, 0.6, 1)
    //   = 0.005183960574111390001176501325136039540388733083636045511
    assert_digits(
        &p("betainc(5/2, 37/10, 3/5, 1)"),
        50,
        "0.005183960574111390001176501325136039540388733083636",
    );
    // mpmath: betainc(0.5, 0.5, 0.3, 1, regularized=True)
    //   = 0.6309898804344546172445694412212634853452756946120199393
    assert_digits(
        &p("betainc_regularized(1/2, 1/2, 3/10, 1)"),
        50,
        "0.63098988043445461724456944122126348534527569461202",
    );
}

#[test]
fn evalf_at_16_and_30_digits() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    // Reference strings are mpmath `nstr(v, 30)` / `nstr(v, 16)` (rounded).
    // mpmath: betainc(0.5, 0.5, 0, 0.3, regularized=True)
    assert_digits(
        &p("betainc_regularized(1/2, 1/2, 0, 3/10)"),
        30,
        "0.369010119565545382755430558779",
    );
    assert_digits(
        &p("betainc_regularized(1/2, 1/2, 0, 3/10)"),
        16,
        "0.3690101195655454",
    );
    // mpmath: betainc(50, 50, 0, 0.55, regularized=True)
    assert_digits(
        &p("betainc_regularized(50, 50, 0, 11/20)"),
        30,
        "0.841347801062901205302347990298",
    );
    assert_digits(
        &p("betainc_regularized(50, 50, 0, 11/20)"),
        16,
        "0.8413478010629012",
    );
    // mpmath: betainc(2.5, 3.7, 0, 0.6)
    assert_digits(
        &p("betainc(5/2, 37/10, 0, 3/5)"),
        30,
        "0.0275434080321464513773697250514",
    );
    assert_digits(&p("betainc(5/2, 37/10, 0, 3/5)"), 16, "0.02754340803214645");
    // mpmath: betainc(0.1, 0.7, 0, 0.5, regularized=True)
    assert_digits(
        &p("betainc_regularized(1/10, 7/10, 0, 1/2)"),
        30,
        "0.894458269131409632799011165627",
    );
    // eval_f64 goes through the same path.
    // mpmath: betainc(1.5, 0.5, 0.2, 0.9) = 0.8853981633974483 (16 digits)
    let v = p("betainc(3/2, 1/2, 1/5, 9/10)").eval_f64().unwrap();
    assert!((v - 0.8853981633974483).abs() < 1e-15, "{v}");
    // x near 0: the value is tiny but carries full relative precision.
    // mpmath: betainc(2.5, 3.7, 0, 1e-6, regularized=True) = 1.22221628443251687197564519441e-14
    assert_digits(
        &p("betainc_regularized(5/2, 37/10, 0, 1/1000000)"),
        30,
        "1.22221628443251687197564519441e-14",
    );
}

#[test]
fn evalf_domain_errors_are_errors_not_panics() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    // a ≤ 0 (non-integer parameters so that no exact fold intervenes).
    assert!(matches!(
        p("betainc(-1/2, 3, 0, 1/2)").eval_f64(),
        Err(SymplexError::Unevaluable { .. })
    ));
    assert!(matches!(
        p("betainc_regularized(3/2, 0, 0, 1/2)").eval_f64(),
        Err(SymplexError::Unevaluable { .. })
    ));
    // limits outside [0, 1]
    assert!(matches!(
        p("betainc(1/2, 3, 0, 2)").eval_f64(),
        Err(SymplexError::Unevaluable { .. })
    ));
    assert!(matches!(
        p("betainc_regularized(1/2, 3, -1/10, 1/2)").eval_f64(),
        Err(SymplexError::Unevaluable { .. })
    ));
    // Reversed limits are the negated integral.
    // mpmath: betainc(0.5, 0.5, 0, 0.3, regularized=True) = 0.36901011956554538…
    let v = p("betainc_regularized(1/2, 1/2, 3/10, 0)")
        .eval_f64()
        .unwrap();
    assert!((v + 0.3690101195655454).abs() < 1e-15, "{v}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Rendering, parsing, macros, backends
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_latex_parse_roundtrip() {
    let ctx = Context::new();
    let (a, b, x, x1) = (
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("x"),
        ctx.symbol("x1"),
    );
    let zero = ctx.int(0);
    let f = x.betainc(&a, &b, &zero);
    let g = x.betainc_regularized(&a, &b, &x1);
    // SymPy argument order in the textual form.
    assert_eq!(format!("{f}"), "betainc(a, b, 0, x)");
    assert_eq!(format!("{g}"), "betainc_regularized(a, b, x1, x)");
    // sympy: latex(betainc(a, b, 0, x)) = \operatorname{B}_{(0, x)}\left(a, b\right)
    assert_eq!(f.to_latex(), r"\operatorname{B}_{(0, x)}\left(a, b\right)");
    // sympy: latex(betainc_regularized(a, b, x1, x2)) = \operatorname{I}_{(x_{1}, x_{2})}\left(a, b\right)
    assert_eq!(g.to_latex(), r"\operatorname{I}_{(x1, x)}\left(a, b\right)");
    roundtrip(&ctx, &f);
    roundtrip(&ctx, &g);
    // Parsing builds the same node as the constructors.
    assert_eq!(ctx.parse("betainc(a, b, 0, x)").unwrap(), f);
    assert_eq!(ctx.parse("betainc_regularized(a, b, x1, x)").unwrap(), g);
    assert_eq!(ctx.parse("BetaInc(a, b, 0, x)").unwrap(), f);
    // Wrong arity is a parse error, not a user function.
    assert!(ctx.parse("betainc(a, b, x)").is_err());
    // MathML falls back to the generic apply form (no wrong glyph).
    let m = f.to_mathml().unwrap();
    assert!(m.contains("betainc"), "{m}");
}

#[test]
fn expr_macro() {
    let ctx = Context::new();
    let (a, b, x, x1) = (
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("x"),
        ctx.symbol("x1"),
    );
    let e = expr!(ctx, betainc(a, b, 0, x));
    assert_eq!(e, x.betainc(&a, &b, &ctx.int(0)));
    let e = expr!(ctx, betainc_regularized(a, b, x1, x));
    assert_eq!(e, x.betainc_regularized(&a, &b, &x1));
    let e = expr!(ctx, betainc_regularized(2, 3, 0, x)).eval();
    assert_eq!(format!("{e}"), "3*x^4 - 8*x^3 + 6*x^2");
}

#[test]
fn backends_report_not_implemented() {
    let ctx = Context::new();
    let (a, b, x) = (ctx.symbol("a"), ctx.symbol("b"), ctx.symbol("x"));
    let f = x.betainc(&a, &b, &ctx.int(0));
    let g = x.betainc_regularized(&a, &b, &ctx.int(0));
    for e in [&f, &g] {
        assert!(
            matches!(e.to_lean(), Err(SymplexError::NotImplemented(_))),
            "{e}: {:?}",
            e.to_lean()
        );
        assert!(e.compile(&["a", "b", "x"]).is_err(), "{e}");
        assert!(e.to_rust_fn("f", &["a", "b", "x"]).is_err(), "{e}");
        assert!(e.to_c_fn("f", &["a", "b", "x"]).is_err(), "{e}");
        assert!(e.to_python().is_err(), "{e}");
    }
}
