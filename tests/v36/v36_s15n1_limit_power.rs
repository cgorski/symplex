//! Tests of session 15 (s15n1): limits of x^g with g tending to 0.
//! Every test says what was wrong before and cites its oracle.
//!
//! Oracle for every value below: mpmath 1.3, the expression evaluated at
//! `x = p ± 10⁻ᵏ` (or `x = 10ᵏ` at `∞`) for `k = 20, 40, 80`, at
//! `dps = 60 + 4k` and `120 + 4k`; the two precisions agree to 30 digits
//! and the three values converge (or diverge like `ln x`).  The values are
//! quoted to 20 digits at `k = 20, 40, 80`.

#![allow(unused_imports)]
use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn lim(ctx: &Context, f: &str, point: &str, dir: Direction) -> Result<Ex, SymplexError> {
    let x = ctx.symbol("x");
    p(ctx, f).try_limit_dir(&x, &p(ctx, point), dir)
}

fn assert_limit(f: &str, point: &str, dir: Direction, expected: &str) {
    let ctx = Context::new();
    let v = lim(&ctx, f, point, dir).unwrap_or_else(|e| panic!("limit of {f} at {point}: {e}"));
    assert_eq!(v, p(&ctx, expected), "limit of {f} at {point}");
}

/// The CI fuzz report on 0.56.0 (`fuzz_calculus`, input `9tSLi4uLCOspAfo=`):
/// `lim_{x→0⁺} x^{e^{√x}}/x` was `0`; it is `1`, as `x^{e^{√x} − 1} =
/// e^{(e^{√x} − 1)·ln x}` and `(e^{√x} − 1)·ln x ≈ √x·ln x → 0`.  Same for
/// `x^{e^{√x} − 1}` (the same expression after canonicalization) and
/// `x^{cosh √x}/x` (exponent `≈ x/2`).  Gruntz rewrote the power as
/// `e^{x + e^{√ω}·ln ω}` (`x = −ln ω` after the move up), whose leading term
/// only the ramified last resort (`ω = u²`) finds; there `x` stayed a
/// constant beside `u`, so the leading term came out `eˣ·ω` — a coefficient
/// of the class of `ω` itself — and the limit `0`.
///
/// Oracle: `x^(exp(sqrt(x)) - 1)` = 0.99999999539482982439,
/// 0.99999999999999999908, 1.0000000000000000000; `x^cosh(sqrt(x))/x` =
/// 0.99999999999999999977, 1.0, 1.0.
#[test]
fn power_with_a_puiseux_exponent_tending_to_zero_is_one() {
    assert_limit("x^(exp(sqrt(x)))/x", "0", Direction::Right, "1");
    assert_limit("x^(exp(sqrt(x)) - 1)", "0", Direction::Right, "1");
    assert_limit("x^(cosh(sqrt(x)))/x", "0", Direction::Right, "1");
}

/// The same class with other rates, bases and points; each was wrong up
/// to 0.56: `x^{e^{∛x} − 1}` → was `0`, `(x²)^{e^{√x} − 1}` → was `0`,
/// `(x − 1)^{e^{√(x−1)} − 1}` at `1⁺` → was `0`, and `x^{1 − e^{−√x}} +
/// x^{2(1 − e^{−√x})}` → was `∞` (it is `2`).
///
/// Oracle: `x^(exp(x^(1/3)) - 1)` = 0.99999007850974735123,
/// 0.9999999999957249387, 1.0; `(x^2)^(exp(sqrt(x)) - 1)` =
/// 0.99999999078965966998, 0.99999999999999999816, 1.0;
/// `(x - 1)^(exp(sqrt(x - 1)) - 1)` at `x = 1 + 10⁻ᵏ` = 0.99999999539482982439,
/// 0.99999999999999999908, 1.0; `x^(1 - exp(-sqrt(x))) + x^(2*(1 -
/// exp(-sqrt(x))))` = 1.9999999861844894957, 1.9999999999999999972, 2.0.
#[test]
fn puiseux_exponents_at_other_rates_bases_and_points() {
    assert_limit("x^(exp(x^(1/3)) - 1)", "0", Direction::Right, "1");
    assert_limit("(x^2)^(exp(sqrt(x)) - 1)", "0", Direction::Right, "1");
    assert_limit("(x - 1)^(exp(sqrt(x - 1)) - 1)", "1", Direction::Right, "1");
    assert_limit(
        "x^(1 - exp(-sqrt(x))) + x^(2*(1 - exp(-sqrt(x))))",
        "0",
        Direction::Right,
        "2",
    );
}

/// Expressions whose limit needs the power's leading term, not only its
/// limit; each was wrong up to 0.56: `(x^{e^{√x}}/x − 1)/(√x·ln x)` → was
/// `∞` (it is `1`: `x^{e^{√x} − 1} − 1 ≈ √x·ln x`), `x^{e^{√x} − 1}·ln x` →
/// was `0` (it is `−∞`), `e^{x^{cosh(x^{2/3}) − 1} − 1}` → was `e^{−1}` (it
/// is `1`), `x^{x^{cos √x − 1}}` → was `1` (it is `0`).
///
/// Oracle: `(x^(exp(sqrt(x)))/x - 1)/(sqrt(x)*log(x))` =
/// 0.99999999774741491031, 0.99999999999999999954, 1.0;
/// `x^(exp(sqrt(x)) - 1)*log(x)` = −46.051701647804989739,
/// −92.103403719761827276, −184.20680743952365472 (`ln x`);
/// `exp(x^(cosh(x^(2/3)) - 1) - 1)` = 1.0, 1.0, 1.0;
/// `x^(x^(cos(sqrt(x)) - 1))` = 9.999999999999999894e−21, 1.0e−40, 1.0e−80.
#[test]
fn leading_terms_of_powers_with_puiseux_exponents() {
    assert_limit(
        "(x^(exp(sqrt(x)))/x - 1)/(sqrt(x)*log(x))",
        "0",
        Direction::Right,
        "1",
    );
    assert_limit("x^(exp(sqrt(x)) - 1)*log(x)", "0", Direction::Right, "-oo");
    assert_limit("exp(x^(cosh(x^(2/3)) - 1) - 1)", "0", Direction::Right, "1");
    assert_limit("x^(x^(cos(sqrt(x)) - 1))", "0", Direction::Right, "0");
}

/// The two-sided limit of `x^{e^{√x} − 1}` at `0` is `1` from both sides
/// (from the left `√x = i·√|x|` and `x^{i·√|x| + …} → 1`).  Up to 0.56 it
/// was refused, "left and right limits differ: left = 1, right = 0" — the
/// right-hand `0` of the test above.
///
/// Oracle: at `x = −10⁻ᵏ`, `0.99999999968584072432 − 4.6051701845570424338e−9·i`,
/// `0.99999999999999999997 − 9.2103403719761827358e−19·i`, `1.0 −
/// 1.8420680743952365472e−38·i`; from the right as above.
#[test]
fn two_sided_limit_of_a_power_with_a_puiseux_exponent() {
    assert_limit("x^(exp(sqrt(x)) - 1)", "0", Direction::Both, "1");
}

/// At `∞`: `x·(x^{e^{x^{−3/2}} − 1} − 1)` → was `−∞`; it is `0`
/// (`x^{e^{x^{−3/2}} − 1} − 1 ≈ x^{−3/2}·ln x`), as is the `cos` variant
/// (≈ `−x^{−3}·ln x/2`), and `x^{3/2}·(x^{e^{x^{−3/2}} − 1} − 1)/ln x` → was
/// `−∞` (it is `1`).  Gruntz's Taylor expansion of the rewritten
/// `ω·e^{x·e^{ω^{3/2}}} − 1` in `ω` (`x` held constant) is `ω·eˣ − 1`,
/// which is 0 (`eˣ = ω⁻¹`); the series engine, given `x` written through
/// `l = ln ω`, took `e^{−l}` for a constant of order 0 and the leading term
/// for `−1`.
///
/// Oracle: `x*(x^(exp(x^(-3/2)) - 1) - 1)` = 4.605170185988091368e−9,
/// 9.2103403719761827361e−19, 1.8420680743952365472e−38;
/// `x*(x^(cos(x^(-3/2)) - 1) - 1)` = −2.302585092994045684e−39,
/// −4.605170185988091368e−79, −9.2103403719761827361e−159;
/// `x^(3/2)*(x^(exp(x^(-3/2)) - 1) - 1)/log(x)` = 1.0, 1.0, 1.0 (to 20 digits).
#[test]
fn powers_with_puiseux_exponents_at_infinity() {
    assert_limit("x*(x^(exp(x^(-3/2)) - 1) - 1)", "oo", Direction::Both, "0");
    assert_limit("x*(x^(cos(x^(-3/2)) - 1) - 1)", "oo", Direction::Both, "0");
    assert_limit(
        "x^(3/2)*(x^(exp(x^(-3/2)) - 1) - 1)/log(x)",
        "oo",
        Direction::Both,
        "1",
    );
}

/// The slow unit of the same CI run (`fuzz_calculus`, input
/// `LIuwCMc9tbWyAIq1tbIAiorKihSKiooUCA==`, 18 s): `series(√(atan(ln x)), x,
/// 0, 3)` has no expansion and stays unevaluated, but took 1.2 s (release)
/// to refuse.  The limit of `x·f` from the left was refused (the side of
/// the cut of `√` is undecided) although it is `0` from the right, so the
/// pole retries `xᵏ·f` (`k = 1..5`) ran, each asking the limit engine for
/// the limits of ever larger derivatives.  Now 2 ms.
///
/// Oracle (exact): there is no power series — `√(atan(ln x)) → i·√(π/2)`
/// as `x → 0⁺`, and its derivative `1/(2x·(1 + ln²x)·√(atan(ln x)))` is
/// unbounded there, so no linear term exists.
#[test]
fn series_without_a_pole_is_refused_without_the_pole_retries() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let start = std::time::Instant::now();
    let s = p(&ctx, "(atan(ln(x)))^(1/2)").series(&x, &ctx.int(0), 3);
    assert!(s.has_unevaluated(), "{s}");
    assert!(start.elapsed().as_secs() < 5, "{:?}", start.elapsed());
}
