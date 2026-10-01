//! Values below the exponent range of the arbitrary-precision floats
//! (about `2^(−2.1·10⁹)`): sums of both signs, and the functions of such
//! values that come back in range.

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `eval_decimal` at 16 and 30 digits refuses `s` as a nonzero number below
/// the exponent range; `eval_f64` is `0.0`, its correctly rounded value.
fn nonzero_underflow(ctx: &Context, s: &str) {
    for digits in [16, 30] {
        match parse(ctx, s).eval_decimal(digits) {
            Err(SymplexError::Unevaluable { reason })
                if reason.contains("is not 0 but underflows") => {}
            other => panic!("{s} at {digits} digits: want a nonzero underflow, got {other:?}"),
        }
    }
    assert_eq!(parse(ctx, s).eval_f64().unwrap(), 0.0, "{s}");
}

/// Before: `0` — the terms underflowed to `0 ± 2^(−2.1·10⁹)` and a sum of
/// such zeros of both signs was a zero to the precision reached.  The
/// values (mpmath, `mp.dps=50`, the same at 80; `y = mpf(10)**10`):
/// `exp(-y) - exp(-2*y)` = `9.27858442032487257807314229893e-4342944820`,
/// `ei(-y) + exp(-y)` = `9.278584419397014136133441e-4342944820`,
/// `log1p(exp(-y))` and `expm1(exp(-y))` = `exp(-y)`,
/// `2*sin(exp(-y)/2)**2` (`1 − cos`) = `4.3046064422547725841e-8685889639`,
/// `mpf(2)**(-y) - mpf(3)**(-y)` = `2.29185980455155474585877784240e-3010299957`
/// (SymPy 1.14's `N` prints the first two and `0` for `log(1 + exp(-y))`).
/// They are held scaled now (`m·2^k`) and certified nonzero.
#[test]
fn sums_of_both_signs_below_the_range_are_not_zero() {
    let ctx = Context::new();
    for s in [
        "exp(-10^10) - exp(-2*10^10)",
        "Ei(-10^10) + exp(-10^10)",
        "log(1 + exp(-10^10))",
        "exp(exp(-10^10)) - 1",
        "cos(exp(-10^10)) - 1",
        "2^(-10^10) - 3^(-10^10)",
        "erfc(10^5) - exp(-10^10)",
        "exp(-10^10) - exp(-10^10 - 10^(-30))",
    ] {
        nonzero_underflow(&ctx, s);
    }
}

/// Before: refused (`PrecisionExhausted`, or "exp overflows" for the
/// reciprocal) — the logarithm of `0 ± underflow` has no bound.  mpmath
/// (`mp.dps=50`, the same at 80; `y = mpf(10)**10`):
/// `log(ei(-y) + exp(-y)) + y` = `-9.9999999995000000001333333332908e-11`,
/// `log(-ei(-y))` = `-10000000023.025850930040456840164915`,
/// `log(erfc(mpf(10)**5))` = `-10000000012.085290407944928507155421`,
/// `exp(-5*mpf(10)**8)` = `1.1178256894171887302861636338441716e-217147241`,
/// `exp(-mpf(10)**9)` = `1.2495342719210132809243784990149911e-434294482`,
/// `log(2*sin(exp(-y)/2)**2)` = `-20000000000.693147180559945309417232`.
#[test]
fn functions_of_values_below_the_range_come_back_in_range() {
    let ctx = Context::new();
    let dec = |s: &str| parse(&ctx, s).eval_decimal(30).unwrap();
    assert_eq!(
        dec("log(Ei(-10^10) + exp(-10^10)) + 10^10"),
        "-9.99999999950000000013333333329e-11"
    );
    assert_eq!(dec("log(-Ei(-10^10))"), "-10000000023.0258509300404568402");
    assert_eq!(dec("log(erfc(10^5))"), "-10000000012.0852904079449285072");
    assert_eq!(
        dec("sqrt(exp(-3*10^9))*exp(10^9)"),
        "1.11782568941718873028616363384e-217147241"
    );
    assert_eq!(
        dec("exp(-3*10^9)^(1/3)"),
        "1.24953427192101328092437849901e-434294482"
    );
    assert_eq!(
        dec("log(1 - cos(exp(-10^10)))"),
        "-20000000000.6931471805599453094"
    );
    // `−10¹⁰ + ln(1 − e^(−10¹⁰))`.
    assert_eq!(dec("log(exp(-10^10) - exp(-2*10^10))"), "-1e10");
}

/// A cancellation of values below the range that is truly 0 is still `0`
/// (`sin²1 + cos²1 = 1`; `(e^(−3·10⁹))^(1/3) = e^(−10⁹)`), found by the zero
/// search at the scale of its terms.
#[test]
fn a_true_zero_below_the_range_is_still_zero() {
    let ctx = Context::new();
    for s in [
        "exp(-10^10)*(sin(1)^2+cos(1)^2) - exp(-10^10)",
        "exp(-3*10^9)^(1/3) - exp(-10^9)",
    ] {
        assert_eq!(parse(&ctx, s).eval_decimal(16).unwrap(), "0", "{s}");
    }
}

/// Before: `0`.  `K₀(10¹⁰)` and `Ai(10⁷)` underflow without a scaled form,
/// so their difference is `0 ± 2^(−2.1·10⁹)` with nothing known of its sign:
/// refused, not printed as 0 (mpmath: `besselk(0, mpf(10)**10) -
/// airyai(mpf(10)**7)` = `1.1628981028123146387e-4342944824`).
#[test]
fn an_undecidable_sum_of_underflows_is_refused() {
    let ctx = Context::new();
    match parse(&ctx, "besselk(0, 10^10) - airyai(10^7)").eval_decimal(16) {
        Err(SymplexError::Unevaluable { reason }) if reason.contains("not known to be 0") => {}
        other => panic!("{other:?}"),
    }
}

/// Before: `solve(exp(−10¹⁰), x)` (and `besselk(0, 10¹⁰)`) was an identity —
/// every `x` a solution: the zero tests read a nonzero number below the
/// exponent range as a zero to the precision reached.
#[test]
fn a_nonzero_constant_below_the_range_is_not_an_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for s in [
        "exp(-10^10)",
        "exp(-10^10) - exp(-2*10^10)",
        "besselk(0, 10^10)",
    ] {
        match parse(&ctx, s).solve(&x) {
            Err(SymplexError::NoSolution { .. }) => {}
            other => panic!("solve({s}, x): {other:?}"),
        }
    }
}
