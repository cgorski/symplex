//! Cancellations at the second order and beyond among values below the
//! exponent range of the arbitrary-precision floats (about `2^(−2.1·10⁹)`).
//! `t` is `exp(−10¹⁰)`, `9.28·10^(−4342944820)`.

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `eval_decimal` at 16 and 30 digits refuses `s` with a reason containing
/// `want`.
fn refused_with(ctx: &Context, s: &str, want: &str) {
    for digits in [16, 30] {
        match parse(ctx, s).eval_decimal(digits) {
            Err(SymplexError::Unevaluable { reason }) if reason.contains(want) => {}
            other => panic!("{s} at {digits} digits: want \"{want}\", got {other:?}"),
        }
    }
}

/// Before: `0` at every number of digits.  Only the first order of a
/// function of a value below the range was kept (`ln(1 + t) = t·(1 +
/// O(t))`), so the first-order terms cancelled into a zero ball that the
/// zero search shrank with the precision, a zero to the precision reached.
/// The values (mpmath 1.3, `mp.dps=30`, the same at 60; `t =
/// exp(-mpf(10)**10)`): `-t**2/2` = `-4.304606442254772584144746e-8685889639`
/// (`log(1+t) - t`, `exp(t) - 1 - t` is its negative), `-t**3/6` =
/// `-1.331355142357840372079138e-13028834458` (`sin(t) - t`), `t**4/24` =
/// `3.088272770450215129662305e-17371779278` (`cos(t) - 1 + t²/2`), `t**2` =
/// `8.609212884509545168289493e-8685889639` (`(1 + t)·t − t`), `-t**2/8` =
/// `-1.076151610563693146036187e-8685889639` (`sqrt(1 + t) − 1 − t/2`); all
/// nonzero, now refused as such.
#[test]
fn second_order_cancellations_below_the_range_are_not_zero() {
    let ctx = Context::new();
    for s in [
        "log(1+exp(-10^10)) - exp(-10^10)",
        "sin(exp(-10^10)) - exp(-10^10)",
        "exp(exp(-10^10)) - 1 - exp(-10^10)",
        "cos(exp(-10^10)) - 1 + exp(-2*10^10)/2",
        "cos(exp(-10^10)) - 1 + exp(-10^10)^2/2",
        "(1+exp(-10^10))*exp(-10^10) - exp(-10^10)",
        "sqrt(1+exp(-10^10)) - 1 - exp(-10^10)/2",
        "log(1+exp(-10^10)) - exp(-10^10) - (cos(exp(-10^10)) - 1)",
    ] {
        refused_with(&ctx, s, "is not 0 but underflows");
        assert_eq!(parse(&ctx, s).eval_f64().unwrap(), 0.0, "{s}");
    }
}

/// Before: refused (`PrecisionExhausted`: the logarithm of a zero ball).
/// The residual is known, so its logarithm is in range.  mpmath 1.3
/// (`mp.dps=30`, the same at 60): `log(t**2/2)` =
/// `-20000000000.69314718055994530941`, `log(t**3/6)` =
/// `-30000000001.79175946922805500081`, `log(t**4/24)` =
/// `-40000000003.17805383034794561965`, `log(t**3/3)` =
/// `-30000000001.09861228866810969139` (`tan t − t`, and `ln(1 + t) − t −
/// (cos t − 1)`), `log(-t**2/8)` = `-20000000002.07944154167983592825 +
/// 3.14159265358979323846264338328j`.
#[test]
fn logarithms_of_the_residuals_are_in_range() {
    let ctx = Context::new();
    let dec = |s: &str| parse(&ctx, s).eval_decimal(30).unwrap();
    assert_eq!(
        dec("log(-(log(1+exp(-10^10)) - exp(-10^10)))"),
        "-20000000000.6931471805599453094"
    );
    assert_eq!(
        dec("log(exp(exp(-10^10)) - 1 - exp(-10^10))"),
        "-20000000000.6931471805599453094"
    );
    assert_eq!(
        dec("log(-(sin(exp(-10^10)) - exp(-10^10)))"),
        "-30000000001.7917594692280550008"
    );
    assert_eq!(
        dec("log(cos(exp(-10^10)) - 1 + exp(-10^10)^2/2)"),
        "-40000000003.1780538303479456196"
    );
    assert_eq!(
        dec("log(tan(exp(-10^10)) - exp(-10^10))"),
        "-30000000001.0986122886681096914"
    );
    assert_eq!(
        dec("log(log(1+exp(-10^10)) - exp(-10^10) - (cos(exp(-10^10)) - 1))"),
        "-30000000001.0986122886681096914"
    );
    assert_eq!(
        dec("log(sqrt(1+exp(-10^10)) - 1 - exp(-10^10)/2)"),
        "-20000000002.0794415416798359283 + 3.14159265358979323846264338328*i"
    );
}

/// Before: `0`.  The series are kept to the fifth order; a cancellation of
/// every kept term leaves only the remainder, and the value is not known
/// to be 0 or not: refused.  (mpmath 1.3, `mp.dps=30`: `-t**7/5040` =
/// `-1.174739381126453784326252e-30400613737`.)
#[test]
fn a_cancellation_beyond_the_kept_orders_is_refused() {
    let ctx = Context::new();
    refused_with(
        &ctx,
        "sin(exp(-10^10)) - exp(-10^10) + exp(-3*10^10)/6 - exp(-5*10^10)/120",
        "not known to be 0",
    );
}

/// A sum below the range that is truly 0 is still `0`: a cancellation of
/// equal monomials (`exp(−10¹⁰)² = exp(−2·10¹⁰)`, `(exp(−3·10⁹))^(1/3) =
/// exp(−10⁹)`) is exact, and one of different values
/// (`exp(−10¹⁰)·(sin²1 + cos²1)`) a zero ball for the zero search.
#[test]
fn true_zeros_below_the_range_stay_zero() {
    let ctx = Context::new();
    for s in [
        "exp(-10^10)*(sin(1)^2+cos(1)^2) - exp(-10^10)",
        "exp(-3*10^9)^(1/3) - exp(-10^9)",
        "exp(-10^10)^2 - exp(-2*10^10)",
        "(1+exp(-10^10))^2 - 1 - 2*exp(-10^10) - exp(-2*10^10)",
    ] {
        assert_eq!(parse(&ctx, s).eval_decimal(16).unwrap(), "0", "{s}");
    }
}
