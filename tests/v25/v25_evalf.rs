//! 0.26 — `evalf` tracks the accuracy of what it computes.
//!
//! Before 0.26 an expression was evaluated once at a fixed working
//! precision and the result returned whatever its accuracy: catastrophic
//! cancellation silently lost digits (`exp(10⁻³⁰) − 1` came out as `0`), and
//! a quotient by a difference that cancels to zero came out as noise
//! presented as a number.  Every value now carries an error bound; the
//! expression is re-evaluated at a higher precision until the requested
//! digits are covered, and otherwise the evaluation is refused with
//! `PrecisionExhausted` (or, for a value that is zero to the precision
//! reached, returns 0).  Reference values: mpmath 1.3 at `mp.dps = 100`
//! with exact rational inputs (`mpf(1)/mpf(10)**30`), by the call quoted.

use symplex::prelude::*;

fn decimal(ctx: &Context, src: &str, digits: u32) -> Result<String, SymplexError> {
    ctx.parse(src).unwrap().eval_decimal(digits)
}

#[test]
fn cancellation_is_recovered_by_re_evaluation() {
    let ctx = Context::new();
    for (src, want) in [
        // mpmath: exp(mpf(1)/mpf(10)**30) - 1 = 1.0000000000000000000000000000005e-30
        ("exp(1/10^30) - 1", "1e-30"),
        // mpmath: log(1 + mpf(1)/mpf(10)**30) = 9.9999999999999999999999999999950e-31
        ("ln(1 + 1/10^30)", "1e-30"),
        // mpmath: sqrt(mpf(10)**40 + 1) - mpf(10)**20 = 4.99999999999999999999999999999999999999987e-21
        ("sqrt(10^40 + 1) - 10^20", "5e-21"),
        // mpmath: cos(mpf(1)/mpf(10)**20) - 1 = -5.0000000000000000000000000000000e-41
        ("cos(1/10^20) - 1", "-5e-41"),
        // exact: 1/10^40
        ("(1 + 1/10^40) - 1", "1e-40"),
    ] {
        assert_eq!(decimal(&ctx, src, 20).unwrap(), want, "{src}");
    }
    let v = ctx.parse("exp(1/10^30) - 1").unwrap().eval_f64().unwrap();
    assert!((v - 1e-30).abs() < 1e-45, "{v}");
}

#[test]
fn a_value_that_is_zero_to_the_precision_is_zero() {
    let ctx = Context::new();
    // sin²1 + cos²1 − 1 is exactly 0; it used to print a rounding residue.
    assert_eq!(decimal(&ctx, "sin(1)^2 + cos(1)^2 - 1", 20).unwrap(), "0");
    assert_eq!(
        ctx.parse("sin(1)^2 + cos(1)^2 - 1")
            .unwrap()
            .eval_f64()
            .unwrap(),
        0.0
    );
}

#[test]
fn dividing_by_a_cancelled_difference_is_refused() {
    let ctx = Context::new();
    for src in [
        "1/(sin(1)^2 + cos(1)^2 - 1)",
        "sign(sin(1)^2 + cos(1)^2 - 1)",
        "floor(3*(sin(1)^2 + cos(1)^2))",
    ] {
        let r = decimal(&ctx, src, 20);
        assert!(
            matches!(r, Err(SymplexError::PrecisionExhausted { .. })),
            "{src}: {r:?}"
        );
    }
    // Rubi's antiderivative of cot(x)/ln(e^(sin x)) divides by
    // sin x − ln(e^(sin x)) ≡ 0; its derivative at x = 13/4 used to evaluate
    // to −64 at every precision, a false "wrong" in the harness self-test.
    let x = ctx.symbol("x");
    let big_f = ctx
        .parse("-ln(sin(x))/(-ln(exp(sin(x))) + sin(x)) + ln(ln(exp(sin(x))))/(-ln(exp(sin(x))) + sin(x))")
        .unwrap();
    for p in [(13, 4), (1, 3)] {
        let r = big_f
            .diff(&x)
            .subs(&x, &ctx.rational(p.0, p.1))
            .eval_decimal(30);
        assert!(
            matches!(r, Err(SymplexError::PrecisionExhausted { .. })),
            "x = {}/{}: {r:?}",
            p.0,
            p.1
        );
    }
}

#[test]
fn well_conditioned_values_are_unchanged() {
    let ctx = Context::new();
    // mpmath: exp(mpf(1)/3)*pi - sqrt(2) = 2.9702321795359994786668909749533
    assert_eq!(
        decimal(&ctx, "exp(1/3)*pi - sqrt(2)", 20).unwrap(),
        "2.9702321795359994787"
    );
    // mpmath: sin(mpf(10)**100) = -0.37237612366127668826208669555316
    assert_eq!(
        decimal(&ctx, "sin(10^100)", 20).unwrap(),
        "-0.37237612366127668826"
    );
    // acosh(-1) = iπ exactly on the branch point of acosh's derivative.
    let z = ctx.int(-1).acosh().eval_complex64().unwrap();
    assert!(
        z.re == 0.0 && (z.im - std::f64::consts::PI).abs() < 1e-15,
        "{z}"
    );
}
