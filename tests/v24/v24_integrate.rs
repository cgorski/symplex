//! 0.25 — rational functions whose resultant has repeated factors, and the
//! budgets behind "no integrand can exhaust memory".
//!
//! Before 0.25 the logarithmic part used the degree-1 member of the
//! remainder sequence for every algebraic factor of the Rothstein–Trager
//! resultant.  A factor of multiplicity `i` needs the member of degree `i`,
//! so `∫ x³/(x⁸ + 1) dx` (`R = (64t² + 1)⁴`) fell through to partial
//! fractions over the complex roots, whose common-denominator expansion
//! passed 8 GB in 3 s (the three Rubi "timeouts" of 0.24).  Each
//! antiderivative here is checked by `F′ = f` at points on both sides of
//! every real pole.

use std::time::{Duration, Instant};

use symplex::prelude::*;

/// `F = ∫ f` is a closed form with `F′ = f` at every `x` in `points`.
fn antiderivative_at(f: &Ex, x: &Ex, points: &[Ex]) -> Ex {
    let big_f = f.integrate(x);
    assert!(!big_f.has_unevaluated(), "∫ {f} dx is unevaluated: {big_f}");
    let df = big_f.diff(x);
    for v in points {
        let (a, b) = (
            df.subs(x, v).eval_complex64().unwrap(),
            f.subs(x, v).eval_complex64().unwrap(),
        );
        assert!(
            (a - b).norm() < 1e-10 * b.norm().max(1.0),
            "∫ {f} dx = {big_f}: F′ = {a} but f = {b} at x = {v}"
        );
    }
    big_f
}

fn points(ctx: &Context) -> Vec<Ex> {
    [(-7, 2), (-1, 3), (1, 5), (5, 4), (3, 1)]
        .iter()
        .map(|&(n, d)| ctx.rational(n, d))
        .collect()
}

#[test]
fn repeated_resultant_factors_integrate_quickly_and_compactly() {
    // SymPy 1.14 `integrate(f, x)` gives the same four closed forms.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts = points(&ctx);
    for (src, want) in [
        ("x^3/(1+x^8)", "1/4*atan(x^4)"),
        ("x^4/(16+x^10)", "1/20*atan(1/4*x^5)"),
        ("x^5/(9+x^12)", "1/18*atan(1/3*x^6)"),
        ("x^7/(x^16+1)", "1/8*atan(x^8)"),
        ("x^3/(x^8+2*x^4+5)", "1/8*atan(1/2*x^4 + 1/2)"),
    ] {
        let f = ctx.parse(src).unwrap();
        let t = Instant::now();
        let big_f = antiderivative_at(&f, &x, &pts);
        assert!(
            t.elapsed() < Duration::from_secs(2),
            "∫ {src}: {:?}",
            t.elapsed()
        );
        assert_eq!(big_f.to_string(), want, "∫ {src}");
    }
}

#[test]
fn quadratic_factors_of_any_multiplicity_get_exact_real_forms() {
    // Unevaluated before 0.25: the log arguments are quadratics (x/(x⁶ + 1),
    // R = (6t − 1)²(36t² + 6t + 1)²) or the factor is quadratic with real
    // roots (1/(x² − 2), a `RootSum` before).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts = points(&ctx);
    for src in [
        "x/(x^8+1)",
        "x/(x^6+1)",
        "x/(x^8+x^4+1)",
        "x^3/(x^12+1)",
        "x^2/(x^9+1)",
        "x^2/(x^6+x^3+1)",
        "x^5/(x^6+x^3+1)",
        "(x^4+1)/(x^6+1)",
        "1/(x^2-2)",
        "x/(x^4-2)",
        "x^3/(x^8-3)",
    ] {
        antiderivative_at(&ctx.parse(src).unwrap(), &x, &pts);
    }
    let big_f = antiderivative_at(&ctx.parse("1/(x^2-2)").unwrap(), &x, &pts);
    assert_eq!(
        big_f.to_string(),
        "-1/4*sqrt(2)*ln(abs(x + sqrt(2))) + 1/4*sqrt(2)*ln(abs(x - sqrt(2)))"
    );
    let big_f = antiderivative_at(&ctx.parse("x/(x^6+1)").unwrap(), &x, &pts);
    assert_eq!(
        big_f.to_string(),
        "1/6*sqrt(3)*atan(1/3*sqrt(3)*(2*x^2 - 1)) - 1/12*ln(x^4 - x^2 + 1) + 1/6*ln(abs(x^2 + 1))"
    );
}

#[test]
fn irreducible_cubic_and_quintic_factors_keep_their_real_roots() {
    // The real root of the resultant's cubic factor contributes
    // α·ln|x − root|; it was dropped before 0.25, so every one of these was
    // rejected by the derivative check and left unevaluated.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts = points(&ctx);
    for src in [
        "1/(x^3-2)",
        "1/(x^3+2)",
        "x/(x^3-2)",
        "1/(x^5-2)",
        "1/(x^6-2)",
    ] {
        antiderivative_at(&ctx.parse(src).unwrap(), &x, &pts);
    }
}

#[test]
fn polynomials_over_quadratics_with_transcendental_coefficients() {
    // Unevaluated before 0.25, after a 1.3 s / 850 000-node search that kept
    // meeting the same 53 sub-integrands.  SymPy 1.14 has closed forms:
    // integrate(x**(3/2)/(-2*x + sin(-1)), x) = -x**(3/2)/3 + sqrt(x)*sin(1)/2
    //   - sqrt(2)*sin(1)**(3/2)*atan(sqrt(2)*sqrt(x)/sqrt(sin(1)))/4, and
    // integrate(atan(sqrt(x)*cos(-4)), x) = x*atan(sqrt(x)*cos(4))
    //   - (2*sqrt(x)/cos(4)**2 - 2*atan(sqrt(x)*cos(4))/cos(4)**3)*cos(4)/2.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts: Vec<Ex> = [(1, 5), (5, 4), (3, 1)]
        .iter()
        .map(|&(n, d)| ctx.rational(n, d))
        .collect();
    for src in [
        "x^(3/2)/(-2*x+sin(-1))",
        "atan(sqrt(x)*cos(-4))",
        "2*x^4/(sin(-1)-2*x^2)",
        "x^3/(pi*x^2+1)",
        "(x^2+1)/(x^2+sqrt(2)*x+3)",
    ] {
        let t = Instant::now();
        antiderivative_at(&ctx.parse(src).unwrap(), &x, &pts);
        assert!(
            t.elapsed() < Duration::from_secs(2),
            "∫ {src}: {:?}",
            t.elapsed()
        );
    }
}

#[test]
fn constants_that_are_zero_are_seen_as_zero() {
    // fuzz_integrate (0.25 development): `∫ ln(x)/(x² + ln 1) dx` took the
    // `atan(x/√a)/√a` route with a = ln 1 = 0, giving F′ = 0.  The integrand
    // is now evaluated first, so it is ∫ ln(x)/x² = −(1 + ln x)/x.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts = points(&ctx);
    for src in [
        "ln(x)/(x^2+ln(1))",
        "ln(x+1)/(x^2+atan(0))",
        "ln(5-x)/(x^2-tan(0))",
        "1/(x^2+sin(pi))",
    ] {
        antiderivative_at(&ctx.parse(src).unwrap(), &x, &pts);
    }
}

#[test]
fn huge_exponents_are_refused_not_expanded() {
    // `expr_to_poly` built a dense polynomial of degree 10⁹.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["1/(x^1000000000+1)", "x^1000000000/(x+1)"] {
        let f = ctx.parse(src).unwrap();
        let t = Instant::now();
        let big_f = f.integrate(&x);
        assert!(
            t.elapsed() < Duration::from_secs(2),
            "∫ {src}: {:?}",
            t.elapsed()
        );
        assert!(big_f.has_unevaluated(), "∫ {src} = {big_f}");
    }
}

#[test]
fn expansions_beyond_the_term_limit_are_left_unexpanded() {
    use symplex::macros::EXPAND_TERM_LIMIT;
    let ctx = Context::new();
    // 2²¹ > 10⁶ terms: a product of 21 binomials.
    let product = (0..21).fold(ctx.int(1), |acc, i| {
        acc * (ctx.symbol(&format!("y{i}")) + 1)
    });
    const { assert!(1usize << 21 > EXPAND_TERM_LIMIT) };
    let t = Instant::now();
    assert_eq!(product.expand(), product);
    // C(104, 4) ≈ 4.6·10⁶ terms: a multinomial power.
    let (a, b, c, d) = (
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("c"),
        ctx.symbol("d"),
    );
    let power = (&a + &b + &c + &d + 1).powi(100);
    assert_eq!(power.expand(), power);
    assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
    // Below the limit nothing changes: (a + b)³ has 4 terms.
    assert_eq!(
        (&a + &b).powi(3).expand(),
        ctx.parse("a^3 + 3*a^2*b + 3*a*b^2 + b^3").unwrap()
    );
}
