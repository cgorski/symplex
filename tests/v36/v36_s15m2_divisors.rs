//! Tests of session 15 (s15m2): proof-carrying divisors in series, limits and Gruntz.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, src: &str) -> Ex {
    ctx.parse(src).unwrap()
}

/// `ln u` and `u^α` of a series whose constant term is a hidden zero have
/// no Taylor series: up to 0.57 the expansion divided by that constant.
/// `series(ln(x + besselj(−1/2, π/2)), x, 0, 4)` was `ln J + x/J −
/// x²/(2J²) + x³/(3J³)` (`J = besselj(−1/2, π/2)`), `series(√(x + J))` was
/// `√J + x/(2√J) − …`, and `limit(ln(x + J), x, 0)` was `ln J`; likewise
/// with `J₁/₂(π)` and with the argument `besselj(−1/2, π/2 + x)` itself.
///
/// Oracle: `J₋₁/₂(z) = √(2/(πz))·cos z` and `J₁/₂(z) = √(2/(πz))·sin z`
/// (DLMF 10.16.1), so `J₋₁/₂(π/2) = J₁/₂(π) = 0` exactly (mpmath
/// `besselj(-0.5, pi/2)` at 50 digits: `0.0` to the working precision);
/// `ln x`, `√x`, `∛x` have no Taylor series at 0, and `ln x` has no
/// two-sided limit there.  A generic constant term keeps its series.
#[test]
fn ln_and_powers_of_a_hidden_zero_constant_term_have_no_series() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    for src in [
        "log(x + besselj(-1/2, pi/2))",
        "sqrt(x + besselj(-1/2, pi/2))",
        "(x + besselj(-1/2, pi/2))^(1/3)",
        "log(x + besselj(1/2, pi))",
        "log(besselj(-1/2, pi/2 + x))",
        "sqrt(besselj(-1/2, pi/2 + x))",
    ] {
        let e = parse(&ctx, src);
        assert!(e.try_series(&x, &zero, 4).is_err(), "series of {src}");
    }
    for src in ["log(x + besselj(-1/2, pi/2))", "log(x + besselj(1/2, pi))"] {
        assert!(
            parse(&ctx, src).try_limit(&x, &zero).is_err(),
            "limit of {src}"
        );
    }
    assert_eq!(
        parse(&ctx, "log(x + 2)")
            .try_series(&x, &zero, 4)
            .unwrap()
            .to_string(),
        "1/24*x^3 - 1/8*x^2 + 1/2*x + ln(2)"
    );
}

/// A piecewise constant term (`H(y)`, `0^y`, `⌊y⌋`) is not a generic
/// nonzero divisor of `ln` or a power: up to 0.57 `series(ln(x + H(y)), x,
/// 0, 4)` was `ln(H(y)) + x/H(y) − x²/(2H(y)²) + x³/(3H(y)³)`,
/// `series(√(x + ⌊y⌋))` was `√⌊y⌋ + x/(2√⌊y⌋) − …` and `series(ln(x +
/// 0^y))` was `ln(0^y) + x/0^y − …`.
///
/// Oracle: at `y = −1` (`H(−1) = 0`, SymPy `Heaviside(-1)` → 0), at `y =
/// 1/2` (`⌊1/2⌋ = 0`) and at `y = 1` (`0^1 = 0`) the functions are `ln x`
/// and `√x`, without a Taylor series at 0.  A parameter that is analytic
/// keeps its generic series: `ln(x + y) = ln y + x/y − x²/(2y²) + x³/(3y³)
/// + …`.
#[test]
fn ln_and_powers_of_a_piecewise_constant_term_have_no_series() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    for src in [
        "log(x + Heaviside(y))",
        "sqrt(x + Heaviside(y))",
        "log(x + 0^y)",
        "sqrt(x + floor(y))",
    ] {
        let e = parse(&ctx, src);
        assert!(e.try_series(&x, &zero, 4).is_err(), "series of {src}");
    }
    assert_eq!(
        parse(&ctx, "log(x + y)")
            .try_series(&x, &zero, 4)
            .unwrap()
            .to_string(),
        "1/3*x^3*y^(-3) - 1/2*x^2*y^(-2) + x/y + ln(y)"
    );
}

/// A limit by substitution never divides by a value with an undecided
/// factor: up to 0.57 `limit((x + 1)/(y·besselj(−1/2, π/2 + x) + x), x, 0)`
/// was `1/(y·besselj(−1/2, π/2))`, and `limit((y·besselj(−1/2, π/2 +
/// x))^(−1/2), x, 0)` was `1/√(y·besselj(−1/2, π/2))` — quotients by 0
/// (`y·J` was taken for nonzero, generic in `y`).
///
/// Oracle: `J₋₁/₂(π/2 + x) = √(2/(π(π/2 + x)))·cos(π/2 + x) = −(2/π)·x +
/// O(x²)` (DLMF 10.16.1), so the first quotient is `(x + 1)/((1 − 2y/π)·x +
/// …)`, which has no two-sided limit at 0 (for `y ≠ π/2`), and the second
/// grows like `|x|^(−1/2)`.  A generic parameter keeps its value:
/// `limit(1/(y·x + 1), x, 0) = 1`.
#[test]
fn limits_by_substitution_do_not_divide_by_a_hidden_zero_factor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    for src in [
        "(x + 1)/(y*besselj(-1/2, pi/2 + x) + x)",
        "(y*besselj(-1/2, pi/2 + x))^(-1/2)",
    ] {
        if let Ok(v) = parse(&ctx, src).try_limit(&x, &zero) {
            assert!(!v.to_string().contains("besselj"), "limit of {src}: {v}");
        }
    }
    assert_eq!(
        parse(&ctx, "1/(y*x + 1)")
            .try_limit(&x, &zero)
            .unwrap()
            .to_string(),
        "1"
    );
}

/// `Π(n | m)` for `n → 0` and a constant `m` is `Σ nʲ·Jⱼ(m)` with a
/// recurrence that divides by `m`: up to 0.57 an `m` that is 0 without
/// looking like it was divided by.  `series(elliptic_pi(x, H(y)), x, 0, 4)`
/// was `x³·((−7/15·H(y)⁻² − …)·E(H(y)) + …)` and the series with `m =
/// besselj(−1/2, π/2)` a polynomial in `1/besselj(−1/2, π/2)`.
///
/// Oracle: at `m = 0` (`H(−1) = 0`; `J₋₁/₂(π/2) = 0`, DLMF 10.16.1)
/// `Π(n | 0) = π/(2√(1 − n))` (DLMF 19.6.3; mpmath `ellippi(0.1, 0)` =
/// `1.6557…` = `pi/(2*sqrt(0.9))`), whose coefficients are finite while
/// those printed had poles at `m = 0`.  A generic `m` keeps its expansion.
#[test]
fn elliptic_pi_is_not_divided_by_a_vanishing_parameter() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    for src in [
        "elliptic_pi(x, Heaviside(y))",
        "elliptic_pi(x, besselj(-1/2, pi/2))",
    ] {
        let e = parse(&ctx, src);
        assert!(e.try_series(&x, &zero, 4).is_err(), "series of {src}");
    }
    let generic = parse(&ctx, "elliptic_pi(x, 1/2)")
        .try_series(&x, &zero, 2)
        .unwrap()
        .to_string();
    assert!(!generic.contains("Series"), "{generic}");
}
