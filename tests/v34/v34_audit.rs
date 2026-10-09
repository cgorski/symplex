//! Audit of the 0.35 numerical evaluation: values below the exponent range
//! (`evalf/extended.rs`) and the zero search of `evaluate_adaptive`.  Each
//! test names what was printed before and the oracle of the true value.
//! `t` is `exp(−10¹⁰)` = `9.27858442032487e-4342944820` (mpmath 1.3,
//! `mp.dps=30`: `exp(-mpf(10)**10)`); the true values of the cancellations
//! below the range are the leading terms of SymPy 1.14 series in `T = t`
//! (`series(besselj(0, T) - 1, T, 0, 4)` = `-T**2/4 + O(T**4)`), evaluated
//! by mpmath with unbounded exponents (`mp.dps=30`: `exp(-mpf(10)**10)**2/4`
//! = `2.15230322112739e-8685889639`).

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// Refused as "not known to be 0" at 16 and 30 digits, `0.0` from `eval_f64`
/// (the correctly rounded value of a number below the range, or of 0).
fn assert_not_known_zero(ctx: &Context, s: &str) {
    for digits in [16, 30] {
        match parse(ctx, s).eval_decimal(digits) {
            Err(SymplexError::Unevaluable { reason }) if reason.contains("not known to be 0") => {}
            other => panic!("{s} at {digits} digits: {other:?}"),
        }
    }
    assert_eq!(parse(ctx, s).eval_f64().unwrap(), 0.0, "{s}");
}

/// Before: `0`.  A function without a rule for values below the exponent
/// range took the placeholder of its argument — `0 ± underflow`, or `p ±
/// underflow` for a value `p + d` such as `exp(t) = 1 + t + …` — and its
/// value was `f(0)` or `f(p)` within an error that shrinks with the
/// precision, so the cancellation against `f(p)` was a zero ball and the
/// lost part `d` was recorded nowhere.  True values (SymPy series in `T`,
/// leading terms): `cos(1)·T` (`sin(exp(T)) − sin 1`, mpmath
/// `cos(1)*exp(-mpf(10)**10)` = `5.01324055749373e-4342944820`), `T/2`
/// (`atan(exp(T)) − π/4`), `−EulerGamma·T` (`gamma(exp(T)) − 1`), `−T²/4`
/// (`besselj(0, T) − 1`), `−2T/√π` (`erfc(T) − 1`, mpmath
/// `-2*exp(-mpf(10)**10)/sqrt(pi)` = `-1.04697613600316e-4342944819`),
/// `T·ln 2` (`2^T − 1`), `T` (`log(2·exp(T)) − log 2`), `−√2·T²/4`
/// (`√(2·cos T) − √2`), `π·ln π·T` (`π^(exp T) − π`), and with `K =
/// besselk(0, 10¹⁰)` (no scaled form, certainly not 0): `−K²/2`
/// (`cos K − 1`), `K` (`exp K − 1`).  Now refused: no precision shows them.
#[test]
fn a_function_of_a_value_below_the_range_keeps_the_lost_part() {
    let ctx = Context::new();
    for s in [
        "sin(exp(exp(-10^10))) - sin(1)",
        "atan(exp(exp(-10^10))) - pi/4",
        "gamma(exp(exp(-10^10))) - 1",
        "besselj(0, exp(-10^10)) - 1",
        "erfc(exp(-10^10)) - 1",
        "2^exp(-10^10) - 1",
        "log(2*exp(exp(-10^10))) - log(2)",
        "sqrt(2*cos(exp(-10^10))) - sqrt(2)",
        "pi^exp(exp(-10^10)) - pi",
        "cos(besselk(0,10^10)) - 1",
        "exp(besselk(0,10^10)) - 1",
    ] {
        assert_not_known_zero(&ctx, s);
    }
}

/// Before: `0`.  A value below the range made a number alone dropped the
/// terms of its polynomial: `log` of it (in range, `−10¹⁰ + …`), a root of a
/// sum of several terms, `abs` of a complex one, and `erf`, which was
/// `2/√π·t·(1 + O(t²))` without its series.  True values (SymPy series in
/// `T`): `T` (`log(T + T²) + 10¹⁰`), `−T²/6` (`log(sin T) + 10¹⁰`),
/// `T^(3/2)/2` (`√(T + T²) − T^(1/2)`), `√2·T⁵/18` (`abs(sin((1 + i)T)) −
/// √2·T`), `T³/6` (`abs(sin(iT)) − abs(iT)`), `−2T³/(3√π)` (`erf(T) −
/// 2T/√π`, mpmath `2*exp(-mpf(10)**10)**3/(3*sqrt(pi))` =
/// `3.00454681328413e-13028834458` in absolute value).
#[test]
fn a_value_below_the_range_made_a_number_keeps_its_smaller_terms() {
    let ctx = Context::new();
    for s in [
        "log(exp(-10^10) + exp(-2*10^10)) + 10^10",
        "log(sin(exp(-10^10))) + 10^10",
        "sqrt(exp(-10^10)+exp(-2*10^10)) - exp(-5*10^9)",
        "sqrt(exp(-10^10)+exp(-2*10^10))/exp(-5*10^9) - 1",
        "abs(sin(exp(-10^10)*(1+I))) - sqrt(2)*exp(-10^10)",
        "abs(sin(I*exp(-10^10))) - abs(I*exp(-10^10))",
        "erf(exp(-10^10)) - 2*exp(-10^10)/sqrt(pi)",
        "erf(exp(-10^10))*sqrt(pi)/2 - exp(-10^10)",
        "erf(exp(-10^10)*(sin(1)^2+cos(1)^2))*sqrt(pi)/2 - exp(-10^10)",
    ] {
        assert_not_known_zero(&ctx, s);
    }
}

/// Unchanged by the above: values stay values — `log(im(sin(i·t)) − t)` =
/// `log(t³/6)` = `−3·10¹⁰ − ln 6` (mpmath `-3*mpf(10)**10 - log(6)` =
/// `-30000000001.79176`), `erf(t)/t` = `2/√π` (mpmath `2/sqrt(pi)` =
/// `1.128379167095513`; `erf` is now its series) — and true zeros stay `0`:
/// `log(t·(sin²1 + cos²1)) + 10¹⁰`, `abs((1 + i)t) − √2·t`,
/// `t/exp(−10¹⁰ + 1)·e − 1`, and `re(atanh(i·t)) − re(i·t)`, `re` now taken
/// on the polynomial: the remainder of an odd series at an imaginary
/// argument is imaginary, so its real part is exactly 0 (without that it
/// would be a sum of underflows of both signs, refused).
#[test]
fn values_and_true_zeros_below_the_range() {
    let ctx = Context::new();
    for (s, want) in [
        (
            "log(im(sin(I*exp(-10^10))) - exp(-10^10))",
            "-30000000001.79176",
        ),
        ("erf(exp(-10^10))/exp(-10^10)", "1.128379167095513"),
        ("log(exp(-10^10)*(sin(1)^2+cos(1)^2)) + 10^10", "0"),
        ("abs(exp(-10^10)*(1+I)) - sqrt(2)*exp(-10^10)", "0"),
        ("exp(-10^10)/exp(-10^10+1)*E - 1", "0"),
        ("re(atanh(I*exp(-10^10))) - re(I*exp(-10^10))", "0"),
        ("re(asin(I*exp(-10^10))) - re(I*exp(-10^10))", "0"),
    ] {
        assert_eq!(parse(&ctx, s).eval_decimal(16).unwrap(), want, "{s}");
    }
}

/// Before: `0`.  The second-order term of a function at an argument with a
/// small part: `f(1 + h) + f(1 − h) − 2f(1) = f″(1)·h²`, `(1 + h)^q − 1 − qh
/// = q(q − 1)/2·h²`.  The canonical form folds `1 + 10⁻³⁰⁰` into one
/// rational, which the zero search held in full at about 1,000 bits — the
/// value needs twice as many; and the exact `10⁷⁰⁰/(10⁷⁰⁰ + 1) = 1 − h +
/// h² − …` was held to its own bits, not to the part `h²` that a
/// cancellation against `1 − 10⁻⁷⁰⁰` leaves.  The order after the first
/// lost term of a series at a small argument (`x⁵/5` of `atan x` once
/// `x³/3` is cancelled) was not recorded either.  mpmath, each the same at
/// two precisions (`mp.dps` 700 and 760, 900 and 960, 2000 and 2100, 2100
/// and 2200): `sin(1+mpf(10)**-300) + sin(1-mpf(10)**-300) - 2*sin(1)` =
/// `-8.4147098480789650665e-601`, `log(1+h) + log(1-h)` = `-1.0e-600`,
/// `cosh(1+h) + cosh(1-h) - 2*cosh(1)` = `1.5430806348152437785e-600` (`h =
/// mpf(10)**-300`), `sqrt(1+h) - 1 - h/2` = `-1.25e-801` (`h =
/// mpf(10)**-400`), `(1+a)*(1+b) - 1 - a - b` =
/// `4.2857142857142857143e-1801` (`a = mpf(3)/7*mpf(10)**-300`, `b =
/// mpf(10)**-1500`), `1/(1+h) - 1 + h` = `1e-1400` (`h = mpf(10)**-700`),
/// `atan(x) - x + x**3/3` = `2.0e-2001` (`x = mpf(10)**-400`).
#[test]
fn second_order_hidden_terms_in_range_are_evaluated() {
    let ctx = Context::new();
    for (s, want) in [
        (
            "sin(1+10^(-300)) + sin(1-10^(-300)) - 2*sin(1)",
            "-8.414709848078965e-601",
        ),
        ("log(1+10^(-300)) + log(1-10^(-300))", "-1e-600"),
        (
            "cosh(1+10^(-300)) + cosh(1-10^(-300)) - 2*cosh(1)",
            "1.543080634815244e-600",
        ),
        ("(1+10^(-400))^(1/2) - 1 - 10^(-400)/2", "-1.25e-801"),
        (
            "(1+(3/7)*10^(-300))*(1+10^(-1500))*(sin(1)^2+cos(1)^2) - 1 - (3/7)*10^(-300) - 10^(-1500)",
            "4.285714285714286e-1801",
        ),
        (
            "(1+10^(-700))^(-1)*(sin(1)^2+cos(1)^2) - 1 + 10^(-700)",
            "1e-1400",
        ),
        // Distributed by the canonical form over `(√2 + √3)²` and `√6`: two
        // exact rationals `10⁷⁰⁰/(5·(10⁷⁰⁰ + 1))` against `1 − 10⁻⁷⁰⁰`.
        (
            "(1+10^(-700))^(-1)*(((sqrt(2)+sqrt(3))^2-2*sqrt(6))/5) - 1 + 10^(-700)",
            "1e-1400",
        ),
        (
            "atan(10^(-400))*(sin(1)^2+cos(1)^2) - 10^(-400) + 10^(-1200)/3",
            "2e-2001",
        ),
    ] {
        assert_eq!(parse(&ctx, s).eval_decimal(16).unwrap(), want, "{s}");
    }
}

/// True zeros of ordinary expressions are still `0` (identities at rational
/// points, Machin's formula, Gamma reflection, and the identities above at
/// arguments with a small part).
#[test]
fn true_zeros_at_rational_points_are_still_zero() {
    let ctx = Context::new();
    for s in [
        "4*atan(1/5) - atan(1/239) - pi/4",
        "atan(7/3) + atan(3/7) - pi/2",
        "log(6) - log(2) - log(3)",
        "gamma(1/3)*gamma(2/3) - 2*pi/sqrt(3)",
        "(sqrt(2)+sqrt(3))^2 - 5 - 2*sqrt(6)",
        "sin(1+10^(-300))*cos(1+10^(-300))*2 - sin(2+2*10^(-300))",
        "(1+10^(-300))^(1/2)*(1+10^(-300))^(1/2) - 1 - 10^(-300)",
        "sin(10^(-400))*(sin(1)^2+cos(1)^2) - sin(10^(-400))",
    ] {
        assert_eq!(parse(&ctx, s).eval_decimal(16).unwrap(), "0", "{s}");
    }
}
