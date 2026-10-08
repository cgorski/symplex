//! Numerical evaluation: a value that is zero to the precision reached is
//! not printed as `0` when a sum lost a term that is certainly not 0 below
//! its error bound.  `t` is `exp(−10¹⁰)`, `9.28·10^(−4342944820)` (mpmath
//! 1.3: `exp(-mpf(10)**10)`).

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// Before: `0` at every number of digits.  The terms in range cancel to a
/// ball around 0 at every precision (`sin²1 + cos²1 = 1`), and the term
/// below the exponent range (or the higher-order term of a series) was
/// absorbed in that ball: the zero search shrank the ball and printed `0`.
/// If the cancelling part is exactly 0 — it is — the value is the lost
/// term: `t` (`sin²1 + cos²1 − 1 + t`), `t²` (`t·(sin²1 + cos²1) − t + t²`),
/// `−t³/6` (`sin(t)·(sin²1 + cos²1) − t`, mpmath `-t**3/6` =
/// `-1.331355142357840372079138e-13028834458`), `cos(1)·t` (`sin(1 + t) −
/// sin 1`), `2·sin(1)·t` (`(sin 1 + t)² + cos²1 − 1`).  Now refused: no
/// precision keeps a term below the exponent range.
#[test]
fn a_zero_ball_that_lost_a_term_below_the_range_is_not_zero() {
    let ctx = Context::new();
    for s in [
        "sin(1)^2+cos(1)^2-1+exp(-10^10)",
        "exp(-10^10)*(sin(1)^2+cos(1)^2)-exp(-10^10)+exp(-2*10^10)",
        "sin(exp(-10^10))*(sin(1)^2+cos(1)^2)-exp(-10^10)",
        "sin(1+exp(-10^10)) - sin(1)",
        "(sin(1)+exp(-10^10))^2+cos(1)^2-1",
        "(sin(1)^2+cos(1)^2-1) + besselk(0, 10^10)",
    ] {
        for digits in [16, 30] {
            match parse(&ctx, s).eval_decimal(digits) {
                Err(SymplexError::Unevaluable { reason })
                    if reason.contains("not known to be 0") => {}
                other => panic!("{s} at {digits} digits: {other:?}"),
            }
        }
        // The `f64` value is 0, correctly rounded.
        assert_eq!(parse(&ctx, s).eval_f64().unwrap(), 0.0, "{s}");
    }
}

/// A term in range lost the same way is kept by evaluating at the
/// precision it needs (here about 6,700 bits; the zero search stopped at
/// 1,024 bits beyond the working precision).  Before: `0`.  mpmath
/// (`mp.dps=2100`, the same at 2200): `sin(1)**2 + cos(1)**2 - 1 +
/// sin(mpf(10)**-2000)` = `1e-2000` to 20 digits, `(sin(1) + h)**2 +
/// cos(1)**2 - 1` with `h = sin(mpf(10)**-2000)` =
/// `1.6829419696157930133e-2000`.
#[test]
fn a_zero_ball_that_lost_a_term_in_range_is_evaluated_deeper() {
    let ctx = Context::new();
    assert_eq!(
        parse(&ctx, "sin(1)^2+cos(1)^2-1+sin(10^(-2000))")
            .eval_decimal(16)
            .unwrap(),
        "1e-2000"
    );
    assert_eq!(
        parse(&ctx, "(sin(1)+sin(10^(-2000)))^2+cos(1)^2-1")
            .eval_decimal(16)
            .unwrap(),
        "1.682941969615793e-2000"
    );
    // Beyond the configured maximum precision (10,000 bits): refused.
    assert!(matches!(
        parse(&ctx, "sin(1)^2+cos(1)^2-1+sin(10^(-5000))").eval_decimal(16),
        Err(SymplexError::PrecisionExhausted { .. })
    ));
}

/// A value that is zero to the precision reached for one of the reasons
/// below is evaluated at the precision that shows what was lost.  Before:
/// `0`.  mpmath (each at two precisions, the same digits): `atan(mpf(10)**
/// -400)*(sin(1)**2+cos(1)**2) - mpf(10)**-400` = `-3.3333333333333333e-1201`
/// at `mp.dps=1300` (the term `x³/3` of `atan` that its value cannot show);
/// `(sqrt(2)+mpf(10)**-400)**2*(sin(1)**2+cos(1)**2) - 2 -
/// 2*sqrt(2)*mpf(10)**-400` = `1.0e-800` (the square of the small part of a
/// sum); `sin(mpf(2)/3+mpf(10)**-400) - sin(mpf(2)/3)` = `7.85887260776948e-401`
/// (the rational `2/3 + 10⁻⁴⁰⁰` and `2/3` round to the same float below
/// 1,331 bits); `cos((sin(1)**2+cos(1)**2-1)+exp(-mpf(920)))-1` =
/// `-3.9547888618583924e-800` (the term `x²/2` of `cos`).
#[test]
fn hidden_terms_in_range_are_evaluated_deeper() {
    let ctx = Context::new();
    for (s, want) in [
        (
            "atan(10^(-400))*(sin(1)^2+cos(1)^2)-10^(-400)",
            "-3.333333333333333e-1201",
        ),
        (
            "(sqrt(2)+10^(-400))^2*(sin(1)^2+cos(1)^2)-2-2*sqrt(2)*10^(-400)",
            "1e-800",
        ),
        ("sin(2/3+10^(-400))-sin(2/3)", "7.85887260776948e-401"),
        (
            "cos((sin(1)^2+cos(1)^2-1)+exp(-920))-1",
            "-3.954788861858392e-800",
        ),
    ] {
        assert_eq!(parse(&ctx, s).eval_decimal(16).unwrap(), want, "{s}");
    }
    // `tan(10⁻⁴⁰⁰⁰) − 10⁻⁴⁰⁰⁰ = 10⁻¹²⁰⁰⁰/3` needs about 40,000 bits,
    // beyond the configured maximum: refused (SymPy 1.14's `N(…, maxn=5000)`
    // gives `0.e-9060`, a zero without digits).
    assert!(matches!(
        parse(&ctx, "tan(10^(-4000)) - 10^(-4000)").eval_decimal(16),
        Err(SymplexError::PrecisionExhausted { .. })
    ));
}

/// True zeros are still `0`: nothing certainly nonzero was lost (a
/// cancellation of terms of one scale, or an exact one).
#[test]
fn true_zeros_are_still_zero() {
    let ctx = Context::new();
    for s in [
        "sin(1)^2+cos(1)^2-1",
        "exp(-10^10)*(sin(1)^2+cos(1)^2) - exp(-10^10)",
        "exp(-10^10)^2 - exp(-2*10^10)",
        "(1+exp(-10^10))^2 - 1 - 2*exp(-10^10) - exp(-2*10^10)",
        "sin(exp(-10^10))*(sin(1)^2+cos(1)^2) - sin(exp(-10^10))",
        "10^(-1500)*(sin(1)^2+cos(1)^2) - 10^(-1500)",
        "atan(10^(-400))*(sin(1)^2+cos(1)^2) - atan(10^(-400))",
    ] {
        assert_eq!(parse(&ctx, s).eval_decimal(16).unwrap(), "0", "{s}");
    }
}
