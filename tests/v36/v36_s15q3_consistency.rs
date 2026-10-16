//! Tests of session 15 (s15q3): canon and eval consistency at zoo, 0^y, parts below the range.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::errors::SymplexError;
use symplex::prelude::*;

fn parse(ctx: &Context, src: &str) -> Ex {
    ctx.parse(src).unwrap()
}

/// `src` as built, and after `eval`.
fn built_and_evaluated(ctx: &Context, src: &str) -> (String, String) {
    let e = parse(ctx, src);
    (e.to_string(), e.eval().to_string())
}

/// The values at `zoo` that `eval` tabulates (`legendre(3, zoo) = zoo`,
/// `polylog(−1, zoo) = 0`, …) are folded when the node is built too, so
/// that `0·f`, `0/f` and `f − f` read the same before and after `eval`.
/// Before: `polylog(−1, zoo)` stayed when built, and `0·polylog(−1, zoo)`
/// was `nan` (`eval`: `0·0 = 0`); `legendre(3, zoo) − legendre(3, zoo)` was
/// `0` (`eval`: `zoo − zoo = nan`); `0/polylog(2, zoo)` was `nan` (`Li₂`
/// tends to `zoo` in every direction: `0`); `0·legendre(−1, zoo)` and
/// `0·jacobi(2, −3, 0, zoo)` were `nan` though both are the constant 1
/// (`eval` gave `0·1 = 0`).
///
/// Oracle: explicit polynomials and DLMF 25.12.12: `P₃(z) = (5z³ − 3z)/2`
/// (`abs → ∞` in every direction), `Li₋₁(z) = z/(1 − z)²` (`→ 0`), `Li₀(z) =
/// z/(1 − z)` (`→ −1`), `Li₂(z) ~ −ln²(−z)/2` (`abs → ∞`), `P₋₁ = P₀ = 1`,
/// `P₂^(−3,0)(x) = 1` (DLMF 18.5.7: the coefficients of `x` and `x²` carry
/// `(−3 + 1 + 2)_m = 0`), `C₃^(−1)(x) = 0` (DLMF 18.5.10).
#[test]
fn values_at_complex_infinity_fold_when_built() {
    let ctx = Context::new();
    for (src, built) in [
        ("polylog(-1, zoo)", "0"),
        ("polylog(0, zoo)", "-1"),
        ("polylog(2, zoo)", "zoo"),
        ("legendre(3, zoo)", "zoo"),
        ("chebyshev_t(-3, zoo)", "zoo"),
        ("betainc(1, 2, 0, zoo)", "zoo"),
        ("gegenbauer(3, -1, zoo)", "0"),
        ("0*polylog(-1, zoo)", "0"),
        ("legendre(3, zoo) - legendre(3, zoo)", "nan"),
        ("0/polylog(2, zoo)", "0"),
        ("0*legendre(-1, zoo)", "0"),
        ("0/jacobi(2, -3, 0, zoo)", "0"),
    ] {
        let (b, ev) = built_and_evaluated(&ctx, src);
        assert_eq!(b, built, "{src} as built");
        assert_eq!(ev, built, "{src} after eval");
    }
    // A polynomial of degree 0 in its variable stays when built, and `eval`
    // takes its constant.
    for (src, value) in [("legendre(-1, zoo)", "1"), ("jacobi(2, -3, 0, zoo)", "1")] {
        let (b, ev) = built_and_evaluated(&ctx, src);
        assert_eq!(b, src, "{src} as built");
        assert_eq!(ev, value, "{src} after eval");
    }
}

/// Where the table has no value at `zoo`, `eval` keeps the node as the
/// constructors do.  Before: `eval` took a closed form there and gave
/// `nan` for `uppergamma(2, zoo)` and `expint(0, zoo)` (no limit: `e^{−z}`
/// grows for `Re z < 0`), `zoo` for `jacobi(1, zoo, 0, 0)` and 0 for
/// `betainc(zoo, 0, 1, 1)`, while the built nodes stayed: `f − f` was 0 when
/// built and `nan` after `eval`, `0·betainc(zoo, 0, 1, 1)` `nan` when built
/// and 0 after.
///
/// Oracle: the construction itself (`canon::unevaluated_at_complex_infinity`:
/// `0·f` is `nan` for a function left at `zoo`; `x − x = 0`).
#[test]
fn untabulated_values_at_complex_infinity_stay_in_eval_too() {
    let ctx = Context::new();
    for src in [
        "uppergamma(2, zoo)",
        "expint(0, zoo)",
        "jacobi(1, zoo, 0, 0)",
        "betainc(zoo, 0, 1, 1)",
    ] {
        let (b, ev) = built_and_evaluated(&ctx, src);
        assert_eq!(b, ev, "{src}: eval keeps the node");
        let e = parse(&ctx, src).eval();
        assert_eq!((&e - &e).eval().to_string(), "0", "{src} − itself");
        assert_eq!((&ctx.zero() * &e).eval().to_string(), "nan", "0·{src}");
        let built = parse(&ctx, &format!("0*({src})"));
        assert_eq!(built.to_string(), "nan", "0·{src} as built");
    }
}

/// `0/0^y` is `0` for `re y < 0` (`0^y = zoo`) and at `y = 0` (`0^0 = 1`),
/// and `nan` elsewhere (`0/0` for `re y > 0`, `0^{it}` has no value): the
/// quotient is the `Piecewise` of the two, never the `0` of one half plane.
/// Before: `0/0^y`, `0·0^(−y)` (the same value), `0/(x·0^y)`, `0/0^(y + 1)`
/// and `y·0/0^y` were `0` (right only for `re y < 0`), and `x + 0/0^y` was
/// `x`.  `x/0^y` itself is unchanged (`cancel` and `series` keep it), and
/// substituting `x = 0` into it gives the same `Piecewise`; `0` times or over
/// that `Piecewise` agrees with the product built at once.
///
/// Oracle: the exact definition of `0^z` (`0` for `re z > 0`, `zoo` for `re
/// z < 0`, `0⁰ = 1`, `nan` on the rest of `re z = 0`; SymPy and mpmath
/// alike), evaluated at `y = −13/10, 13/10, 0, i`.
#[test]
fn zero_over_zero_power_is_piecewise() {
    let ctx = Context::new();
    let y = ctx.symbol("y");
    let x = ctx.symbol("x");
    let over = "Piecewise(0 if 0 > re(y) | y == 0, nan if True)";
    for src in ["0/0^y", "0*0^(-y)", "0/(x*0^y)", "y*0/0^y", "0*(0/0^y)"] {
        let e = parse(&ctx, src);
        assert_eq!(e.to_string(), over, "{src}");
        for (point, value) in [("-13/10", "0"), ("13/10", "nan"), ("0", "0"), ("I", "nan")] {
            let at = e.subs(&y, &parse(&ctx, point)).eval();
            assert_eq!(at.to_string(), value, "{src} at y = {point}");
            let direct = parse(&ctx, &src.replace('y', &format!("({point})"))).eval();
            assert_eq!(direct.to_string(), value, "{src} with y = {point} first");
        }
    }
    assert_eq!(
        parse(&ctx, "0/0^(y + 1)").to_string(),
        "Piecewise(0 if 0 > re(y) + 1 | y + 1 == 0, nan if True)"
    );
    assert_eq!(parse(&ctx, "x + 0/0^y").to_string(), format!("x + {over}"));
    assert_eq!(parse(&ctx, "0/(0/0^y)").to_string(), "nan");
    let q = parse(&ctx, "x/0^y");
    assert_eq!(q.cancel(&x).to_string(), "x/0^y");
    assert_eq!(q.series(&x, &ctx.zero(), 3).to_string(), "x/0^y");
    assert_eq!(q.subs(&x, &ctx.zero()).to_string(), over);
    // A generic denominator stays 0 (`0^y + 1` is never 0).
    assert_eq!(parse(&ctx, "0/(0^y + 1)").to_string(), "0");
}

/// A part of a complex value that the route computes on its own is shown
/// with its digits however small next to the other part: `Ei(z) = ±iπ −
/// E₁(−z)` has the real part `−Re E₁(−z)`, `γ(a, z) = Γ(a) − Γ(a, z)` (real
/// `a`) the imaginary part `−Im Γ(a, z)`.  Before: one error bound
/// relative to the modulus held both parts, and `Ei(−1000 + i)` printed
/// `3.14159265358979323846264338328*i` at 30 digits (its real part is
/// `−2.7·10⁻⁴³⁸`), `Ei(−10⁹ + i)` likewise, `lowergamma(−1/2, 1000 + i)`
/// printed `−3.54490770181103205459633496668` (its imaginary part is
/// `1.3·10⁻⁴³⁹`).
///
/// Oracle: mpmath 1.3 at 45 digits: `ei(mpc(-1000, 1))` =
/// `-2.73554972804168525095280628542970846524145053e-438 +
/// 3.14159265358979323846264338327950288419716940j`, `ei(mpc(-10**9, 1))` =
/// `-6.75126246653617503380584119088258385234980601e-434294492 + …j`,
/// `gamma(-0.5) - gammainc(-0.5, mpc(1000, 1))` =
/// `-3.54490770181103205459633496668229036559509891 +
/// 1.34996898988844198779961140714420530981700797e-439j` (the same at 60
/// digits), `li(mpc(0, 10**-300))` =
/// `3.28238833864548825797719954497350359117975487e-306 + 3.14159…j`.
#[test]
fn a_small_part_next_to_a_large_one_keeps_its_digits() {
    let ctx = Context::new();
    for (src, re, im) in [
        (
            "Ei(-1000 + I)",
            "-2.73554972804168525095280628543e-438",
            "3.14159265358979323846264338328",
        ),
        (
            "Ei(-10^9 + I)",
            "-6.75126246653617503380584119088e-434294492",
            "3.14159265358979323846264338328",
        ),
        (
            "Ei(-1000 - I)",
            "-2.73554972804168525095280628543e-438",
            "-3.14159265358979323846264338328",
        ),
        (
            "lowergamma(-1/2, 1000 + I)",
            "-3.54490770181103205459633496668",
            "1.34996898988844198779961140714e-439",
        ),
        (
            "li(10^(-300)*I)",
            "3.28238833864548825797719954497e-306",
            "3.14159265358979323846264338328",
        ),
    ] {
        let d = parse(&ctx, src).eval_decimal(30).unwrap();
        let expected = match im.strip_prefix('-') {
            Some(abs) => format!("{re} - {abs}*i"),
            None => format!("{re} + {im}*i"),
        };
        assert_eq!(d, expected, "{src}");
    }
}

/// A part below the exponent range next to a part within it is refused by
/// the route that computes it on its own, as a whole value below the range
/// is, never printed 0.  Before: `Ei(−10¹⁰ + i)` printed
/// `3.14159265358979323846264338328*i` (its real part is
/// `−5.01·10^(−4342944830)`), `lowergamma(−1/2, 10¹⁰ − i)` printed
/// `−3.54490770181103205459633496668` (its imaginary part is `−Im Γ(−1/2,
/// 10¹⁰ − i)`, of the size `10^(−4342944834)`).  The refusal is the route's:
/// a sum holding the value is refused too (`1 + Ei(−10¹⁰ + i)`).
///
/// Oracle: mpmath 1.3 `ei(mpc(-10**10, 1))` =
/// `-5.01324055621163514678329480425828733879977393e-4342944830 +
/// 3.14159…j`, `gammainc(-0.5, mpc(10**10, -1))` of modulus
/// `10^(−4342944834)`; both below `2^(−2³¹) ≈ 10^(−6.46·10⁸)`.
#[test]
fn a_part_below_the_range_next_to_one_within_it_is_refused() {
    let ctx = Context::new();
    for src in [
        "Ei(-10^10 + I)",
        "Ei(-10^10 - I/10^5)",
        "lowergamma(-1/2, 10^10 - I)",
        "lowergamma(-1/2, 10^10 + I/10^5)",
        "1 + Ei(-10^10 + I)",
    ] {
        let d = parse(&ctx, src).eval_decimal(30);
        assert!(
            matches!(&d, Err(SymplexError::Unevaluable { reason })
                if reason.contains("lies below the arbitrary-precision exponent range")),
            "{src}: {d:?}"
        );
    }
}

/// `Max`/`Min` of an argument certified non-real is `nan` (0.59); the
/// certificate is an evaluation, now memoised: rebuilding the node (each
/// `eval`, `subs`) reads the memo.  The answers are those of the first
/// evaluation.
///
/// Oracle: mpmath 1.3 `zeta(mpc(3/7., 1))` = `0.107500980436046 −
/// 0.675325079763582j` (nonzero imaginary part), `besselj(1/3., 2)` =
/// `0.442939818148576` (real).
#[test]
fn min_max_non_real_check_is_memoised() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "Max(x, zeta(3/7 + I))");
    for _ in 0..3 {
        assert_eq!(e.subs(&x, &ctx.int(1)).eval().to_string(), "nan");
        assert_eq!(parse(&ctx, "Max(1, zeta(3/7 + I))").to_string(), "nan");
    }
    let real = parse(&ctx, "Min(x, besselj(1/3, 2))");
    for _ in 0..3 {
        assert_eq!(real.eval().to_string(), "min(x, besselj(1/3, 2))");
    }
}
