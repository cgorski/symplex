//! "The program shows a wrong value" (0.32): rewrite routes on expressions
//! whose denominator vanishes identically, and decimal values below the
//! exponent range.  Each test says what was wrong before and cites its
//! oracle: SymPy 1.14 (`sympy.<route>(...)`), or mpmath 1.3 at two working
//! precisions that agree.

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap_or_else(|e| panic!("parse {s}: {e:?}"))
}

fn underflow_refused(ctx: &Context, s: &str) {
    for digits in [16, 30] {
        match parse(ctx, s).eval_decimal(digits) {
            Err(SymplexError::Unevaluable { reason }) if reason.contains("underflows") => {}
            other => panic!("{s} at {digits} digits: want an underflow error, got {other:?}"),
        }
    }
}

// ── Identically vanishing denominators ──────────────────────────────────────

/// Before: `0` — the `expand` strategy multiplied the numerator out to 0
/// and kept `0·(…)`, while `together` and `ratsimp` gave `nan`; the
/// denominator is 0 as a rational function, so the quotient is `0/0`.
/// SymPy: simplify((x*(x+1)-x**2-x)/(-x/(x+1)+x*(-x/(x+1)+1))) -> nan
/// (its `cancel` and `ratsimp` give 0).
#[test]
fn simplify_of_zero_over_a_vanishing_denominator_is_nan() {
    let ctx = Context::new();
    let e = parse(&ctx, "(x*(x+1)-x^2-x)/(-x/(x+1)+x*(-x/(x+1)+1))");
    assert_eq!(e.simplify().to_string(), "nan");
    assert_eq!(e.together().to_string(), "nan");
    assert_eq!(e.ratsimp().to_string(), "nan");
    // SymPy: simplify(x/(x*(x+1)-x**2-x)) -> zoo*x (zoo: `x/0` is `zoo` here).
    assert_eq!(
        parse(&ctx, "x/(x*(x+1)-x^2-x)").simplify().to_string(),
        "zoo"
    );
}

/// Before: `together` returned both quotients unchanged (the first) or as
/// `(x·D + 1)/D` over the vanishing `D` (the second), while `ratsimp` gave
/// `nan` and `zoo`.  SymPy: together((x*(x+1)-x**2-x)/(x*(x+2)-x**2-2*x))
/// -> nan; together(x + 1/(x*(x+1)-x**2-x)) -> x + zoo.
#[test]
fn together_sees_a_denominator_that_vanishes_once_multiplied_out() {
    let ctx = Context::new();
    let e = parse(&ctx, "(x*(x+1)-x^2-x)/(x*(x+2)-x^2-2*x)");
    assert_eq!(e.together().to_string(), "nan");
    assert_eq!(e.ratsimp().to_string(), "nan");
    assert_eq!(e.simplify().to_string(), "nan");
    // SymPy: together(x/(x*(x+1)-x**2-x)) -> zoo*x
    assert_eq!(
        parse(&ctx, "x/(x*(x+1)-x^2-x)").together().to_string(),
        "zoo"
    );
    assert_eq!(
        parse(&ctx, "x + 1/(x*(x+1)-x^2-x)").together().to_string(),
        "zoo"
    );
    // A denominator that does not vanish is untouched.
    let f = parse(&ctx, "(x*(x+1)-x^2-x)/(x*(x+2)-x^2-x)");
    assert_eq!(f.together(), f);
}

/// Before: `nan` — `expand` distributed the `zoo` of the inner `1/0` over
/// `(x + 3)`, and `zoo·x + zoo` is `nan`; in the arithmetic of `1/0 = zoo`
/// that `ratsimp`, `together` and substitution follow, `a/(b + zoo)` is 0.
/// SymPy: simplify(abs(x-3)/(cos(x)+2 + (x+3)/((x+y)**2 - x**2 - 2*x*y -
/// y**2))) -> 0; simplify(x/(1+1/(x*(x+1)-x**2-x))) -> 0.
#[test]
fn an_absorbed_zero_denominator_simplifies_as_ratsimp_says() {
    let ctx = Context::new();
    let e = parse(
        &ctx,
        "abs(x-3)/(cos(x)+2 + (x+3)/((x+y)^2 - x^2 - 2*x*y - y^2))",
    );
    assert_eq!(e.simplify().to_string(), "0");
    assert_eq!(e.ratsimp().to_string(), "0");
    let e = parse(&ctx, "x/(1+1/(x*(x+1)-x^2-x))");
    assert_eq!(e.simplify().to_string(), "0");
}

/// Before: `ratsimp` gave `zoo` — `√(x(x + 1) − x² − x)` is a generator
/// to it, nonzero as an indeterminate, but it is `√0 = 0`.
/// SymPy: ratsimp(sqrt(x*(x+1)-x**2-x)/(x*(x+2)-x**2-2*x)) -> nan.
#[test]
fn ratsimp_of_a_generator_that_is_zero_over_zero_is_nan() {
    let ctx = Context::new();
    let e = parse(&ctx, "sqrt(x*(x+1)-x^2-x)/(x*(x+2)-x^2-2*x)");
    assert_eq!(e.ratsimp().to_string(), "nan");
    assert_eq!(e.simplify().to_string(), "nan");
    assert_eq!(e.together().to_string(), "nan");
}

/// Before: `zoo` — the trigonometric rules turn the denominator into 0,
/// and the canonical `N·0⁻¹` is `zoo` for a numerator that is not 0 on
/// its face; this numerator vanishes too, so the quotient is `0/0`.
/// SymPy: simplify((x*(x+1)-x**2-x)/(sin(x)**2+cos(x)**2-1)) -> 0 (its
/// `is_zero` makes the numerator 0 before the denominator is looked at;
/// judged: the principled value of `0/0` is `nan`, as `together` gives
/// for a rational `0/0`).
#[test]
fn zero_over_a_trigonometric_zero_is_nan() {
    let ctx = Context::new();
    let e = parse(&ctx, "(x*(x+1)-x^2-x)/(sin(x)^2+cos(x)^2-1)");
    assert_eq!(e.simplify().to_string(), "nan");
}

/// Before: `zoo` from `rationalize_denom` and `simplify`: the conjugate
/// `√(x²) + x` of `√(x²) − x` makes the denominator `x² − x² = 0`, but
/// both factors vanish only on half-planes, and the quotient is `1/4` at
/// `x = −2`.  SymPy: radsimp(1/(sqrt(x**2)-x)) -> 1/(-x + sqrt(x**2));
/// radsimp(1/(sqrt(x)*sqrt(x+1)-sqrt(x*(x+1)))) -> unchanged;
/// (1/(sqrt(x**2)-x)).subs(x,-2) -> 1/4;
/// (1/(sqrt(x)*sqrt(x+1)-sqrt(x*(x+1)))).subs(x,-2) -> -sqrt(2)/4.
#[test]
fn a_conjugate_that_annihilates_the_denominator_is_not_used() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m2 = ctx.int(-2);
    for (s, want) in [
        ("1/(sqrt(x^2) - x)", 0.25),
        (
            "1/(sqrt(x)*sqrt(x+1) - sqrt(x*(x+1)))",
            -std::f64::consts::SQRT_2 / 4.0,
        ),
    ] {
        let e = parse(&ctx, s);
        for r in [e.rationalize_denom(), e.simplify()] {
            let v = r.subs(&x, &m2).eval_f64().unwrap();
            assert!((v - want).abs() < 1e-15, "{s} -> {r} = {v} at x = -2");
        }
    }
}

// ── Values below the exponent range ─────────────────────────────────────────

/// Before: `0` at every precision — the value is about 10^(−4342944825),
/// below astro-float's exponent range, and the zero policy printed the
/// underflow as a zero to the precision reached; 0 is not a digit of a
/// positive number.  mpmath: mp.dps=30; erfc(mpf(10)**5) =
/// 5.23488067975404550060114357077e-4342944825 (dps 40 agrees);
/// exp(-mpf(10)**10) = 9.27858442032487257807314229893e-4342944820;
/// besselk(0, mpf(10)**10) = 1.16289810281231463874435691794e-4342944824.
#[test]
fn a_nonzero_value_below_the_exponent_range_is_refused() {
    let ctx = Context::new();
    for s in [
        "erfc(10^5)",
        "exp(-10^10)",
        "erfc(10^5)*2",
        "besselk(0, 10^10)",
        "-exp(-10^10)",
        "exp(-10^10)*I",
        "erfc(10^5)^2",
        "sin(exp(-10^10))",
        // Terms of one sign.  mpmath: airyai(mpf(10)**7) + erfc(mpf(10)**5)
        // = 5.23488067975404550060114357077e-4342944825.
        "airyai(10^7) + erfc(10^5)",
        // Before: `PrecisionExhausted` (the bound of an `exp` that came out
        // 0 was unknown).  mpmath: exp(-mpf(10)**10 - sqrt(2)) =
        // 2.25577914444155945160638169735e-4342944820.
        "exp(-10^10 - sqrt(2))",
    ] {
        underflow_refused(&ctx, s);
    }
    // `f64` is the correctly rounded value, 0.
    assert_eq!(parse(&ctx, "erfc(10^5)").eval_f64().unwrap(), 0.0);
    assert_eq!(parse(&ctx, "exp(-10^10)").eval_f64().unwrap(), 0.0);
}

/// A value that is 0, or only 0 to the precision reached, still prints as
/// `0`; a tiny part next to a normal one is negligible; the sign of an
/// underflow is still undecided.
#[test]
fn zeros_and_negligible_parts_are_unchanged_by_the_underflow_rule() {
    let ctx = Context::new();
    let dec = |s: &str| parse(&ctx, s).eval_decimal(16);
    assert_eq!(dec("erfc(10^5) - erfc(10^5)").unwrap(), "0");
    // Truly 0 (sin²1 + cos²1 = 1): a cancellation, not a nonzero number.
    assert_eq!(
        dec("exp(-10^10)*(sin(1)^2+cos(1)^2) - exp(-10^10)").unwrap(),
        "0"
    );
    assert_eq!(dec("exp(-10^10) + 1").unwrap(), "1");
    assert_eq!(dec("1 + I*exp(-10^10)").unwrap(), "1");
    assert!(matches!(
        dec("sign(erfc(10^5))"),
        Err(SymplexError::PrecisionExhausted { .. })
    ));
}
