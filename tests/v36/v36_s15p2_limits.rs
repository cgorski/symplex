//! Tests of session 15 (s15p2): limits and series: hidden zeros in functions of parameters, piecewise leading terms, slow refusals.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

use crate::perf_assert::assert_fast;
use std::time::Duration;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn lim(ctx: &Context, f: &str, point: &str, dir: Direction) -> Result<String, SymplexError> {
    let x = ctx.symbol("x");
    p(ctx, f)
        .try_limit_dir(&x, &p(ctx, point), dir)
        .map(|v| v.to_string())
}

/// A hidden zero inside a function of a parameter is no nonzero value: the
/// parametric zero tests split off symbol-free *factors* only, so
/// `cos(y·J + π/2) = −sin(y·J)` with `J = besselj(−1/2, π/2)` read as
/// nonzero for a generic `y`.  Up to 0.58 the limit of `tan(π/2 +
/// y·besselj(−1/2, π/2 + x))` at 0 was `tan(y·besselj(−1/2, π/2) +
/// π/2)` (a value at a pole), as was the limit of `tan(π/2 + y·J + x)`;
/// the limit of `x/(x + sin(y·J))` from the right was `Piecewise((1, y =
/// 0), (1, y = π/J), (0, True))`, the limit of `1/(x + sin(y·J))` was
/// `1/sin(y·J)` and its series `x²/sin³(y·J) − x/sin²(y·J) +
/// 1/sin(y·J)`.  Now refused.
///
/// Oracle: `J₋₁/₂(z) = √(2/(πz))·cos z` (DLMF 10.16.1), so `J = 0` and
/// `J₋₁/₂(π/2 + x) = −(2/π)·x + O(x²)`: `tan(π/2 + y·J₋₁/₂(π/2 + x)) =
/// −cot(y·J₋₁/₂(π/2 + x))` is unbounded at `x = 0` for every `y ≠ 0`
/// (mpmath, dps 80, `y = 7/10`: `±2.24399475256414·10²⁰` at `x = ±10⁻²⁰`),
/// and `x/(x + sin(y·J)) = 1` for every `y`.  A nonzero constant keeps its
/// generic value: `J₋₁/₂(1) = √(2/π)·cos 1 = 0.431098868018376`.
#[test]
fn hidden_zero_inside_a_function_of_a_parameter_is_not_a_nonzero_value() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    assert!(
        lim(
            &ctx,
            "tan(pi/2 + y*besselj(-1/2, pi/2 + x))",
            "0",
            Direction::Right
        )
        .is_err()
    );
    assert!(
        lim(
            &ctx,
            "tan(pi/2 + y*besselj(-1/2, pi/2) + x)",
            "0",
            Direction::Both
        )
        .is_err()
    );
    for f in [
        "x/(x + sin(y*besselj(-1/2, pi/2)))",
        "1/(x + sin(y*besselj(-1/2, pi/2)))",
    ] {
        assert!(lim(&ctx, f, "0", Direction::Right).is_err(), "{f}");
    }
    let s = p(&ctx, "1/(x + sin(y*besselj(-1/2, pi/2)))").series(&x, &zero, 3);
    assert!(s.has_unevaluated(), "{s}");
    assert!(
        lim(
            &ctx,
            "x/(x + sin(y*besselj(-1/2, 1)))",
            "0",
            Direction::Right
        )
        .is_ok(),
        "a nonzero constant keeps its generic limit"
    );
}

/// A hidden zero that `hidden_zero` proves is replaced by 0 inside the
/// function: `y·asinh 2 − y·ln(2 + √5)` is `y·0`.  Up to 0.58
/// `limit(1/(x + sin(y·(asinh 2 − ln(2 + √5)))), x, 0, +)` was the value
/// `1/sin(y·(asinh 2 − ln(2 + √5)))` (a quotient by 0), and the series a
/// Taylor series in its powers; the function is `1/x`.
///
/// Oracle (exact): `asinh 2 = ln(2 + √5)` (DLMF 4.37.19), so the function
/// is `1/x`, whose limit from the right is `+∞`.
#[test]
fn a_proved_hidden_zero_inside_a_function_is_substituted() {
    let ctx = Context::new();
    assert_eq!(
        lim(
            &ctx,
            "1/(x + sin(y*(asinh(2) - log(2 + sqrt(5)))))",
            "0",
            Direction::Right
        )
        .unwrap(),
        "oo"
    );
}

/// A step function at an argument nonzero for generic parameters is
/// continuous there: its value is the limit, with the exact condition.  Up
/// to 0.58 these limits were refused once the zero tests read `H(y)` as a
/// coefficient that may vanish on a region (0.55: `Piecewise((H(y), y ≠
/// 0))`): `limit(H(x + y), x, 0)`, `limit(H(x + y) + x, x, 0)`.
///
/// Oracle (exact): for a real `y ≠ 0`, `H(x + y) = H(y)` for `|x| < |y|`;
/// at `y = 0`, `H(x)` has the one-sided limits 0 and 1 (no two-sided
/// limit, right limit 1); `H(x + y)` at `x → 1` jumps at `y = −1`.  SymPy
/// 1.14: `limit(Heaviside(x + y), x, 0)` = `Heaviside(y)`.
#[test]
fn a_step_at_a_generic_argument_is_its_value_with_the_condition() {
    let ctx = Context::new();
    assert_eq!(
        lim(&ctx, "Heaviside(x + y)", "0", Direction::Both).unwrap(),
        "Piecewise(H(y) if y != 0)"
    );
    assert_eq!(
        lim(&ctx, "Heaviside(x + y)", "0", Direction::Right).unwrap(),
        "Piecewise(1 if y == 0, H(y) if y != 0)"
    );
    assert_eq!(
        lim(&ctx, "Heaviside(x + y) + x", "0", Direction::Both).unwrap(),
        "Piecewise(H(y) if y != 0)"
    );
    assert_eq!(
        lim(&ctx, "Heaviside(x + y)", "1", Direction::Both).unwrap(),
        "Piecewise(H(y + 1) if y + 1 != 0)"
    );
    // A step in a divisor stays refused: `x/min(y, x, 5)` is 1 for `y > 0`
    // and tends to 0 for `y < 0`.
    assert!(lim(&ctx, "x/Min(y, x, 5)", "0", Direction::Both).is_err());
}

/// Gruntz's work budget counts the zero tests of coefficients.  Up to 0.58
/// `limit(E1(digamma(x))^Ei(x), x, 0, −)` ran for over 25 s (release): the
/// fallback expansion tested coefficients that are polynomials in `ζ(k)`
/// and `γ` with a thousand distinct nodes, each `ζ` evaluated at 166 bits,
/// none of it charged.  Now refused within the budget (1.5 s release).
///
/// Oracle: none needed for a refusal; the time is the claim.
#[test]
fn gruntz_charges_the_evaluations_of_its_coefficients() {
    let ctx = Context::new();
    let r = assert_fast!(
        Duration::from_secs(120),
        lim(&ctx, "E1(digamma(x))^Ei(x)", "0", Direction::Left)
    );
    assert!(r.is_err(), "{r:?}");
}

/// Slow refusals of series.  Up to 0.58 `series(x⁵·√(atan(ln x)), x, 0, 8)`
/// took 0.7 s (release) to stay unevaluated: `atan(ln x)` went to the
/// differentiation fallback, whose limits of derivatives each ran Gruntz;
/// now `f(ln x)` is the constant `f(l)` of the log-extended expansion
/// (3 ms).  `series(x^(−N) + eˣ, x, 0, 3)`, `sin(x)/x^N`, `1/sin(x)^N` (`N =
/// 65537`), `series(sin x, x, 0, 2³¹)` and `series_at_infinity(sin(1/x),
/// x, 65537)` ran for hours with little memory: an expansion needing more
/// than 1,024 terms is refused at once.
///
/// Oracle (exact): `√(atan(ln x)) → i·√(π/2)` as `x → 0⁺` with derivative
/// `1/(2x·(1 + ln²x)·√(atan(ln x)))`, unbounded: no power series, and the
/// two-sided expansion has the coefficient `√(atan(ln|x|))` of `x⁵`.
#[test]
fn slow_series_refusals_are_fast() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let s = assert_fast!(
        Duration::from_secs(10),
        p(&ctx, "x^5*sqrt(atan(log(x)))").series(&x, &zero, 8)
    );
    assert!(s.has_unevaluated(), "{s}");
    for src in ["x^(-65537) + exp(x)", "sin(x)/x^65537", "1/sin(x)^65537"] {
        let s = assert_fast!(Duration::from_secs(10), p(&ctx, src).series(&x, &zero, 3));
        assert!(s.has_unevaluated(), "{src}: {s}");
    }
    let s = assert_fast!(
        Duration::from_secs(10),
        p(&ctx, "sin(x)").series(&x, &zero, 1 << 31)
    );
    assert!(s.has_unevaluated(), "{s}");
    let s = assert_fast!(
        Duration::from_secs(10),
        p(&ctx, "sin(1/x)").series_at_infinity(&x, 65537)
    );
    assert!(s.has_unevaluated(), "{s}");
    // Within the bound the expansion is made.
    assert_eq!(
        p(&ctx, "x^(-20) + exp(x)").series(&x, &zero, 3).to_string(),
        "1/2*x^2 + x + x^(-20) + 1"
    );
}
