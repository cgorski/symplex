//! Tests of session 15 (s15g1): undefined values are not folded away.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, src: &str) -> Ex {
    ctx.parse(src).unwrap()
}

/// The library functions that are infinite at `−∞` (and `E` at `±∞`)
/// fold to their values when they are built, so `0·f(−∞)` is `0·∞`.  Up
/// to 0.52 they stayed unevaluated and `0·besseli(1, −∞)`,
/// `0·besselk(1, −∞)`, `0·expint(1, −∞)`, `0·uppergamma(2, −∞)`,
/// `0·elliptic_e(∞)`, `0·elliptic_e(−∞)` and `0·Chi(−∞)` were 0.  A value
/// is a directed infinity where the limit has one (with its finite part
/// where that is exact, as `ln(−∞) = ∞ + iπ`), `zoo` otherwise.  Rebuilding
/// the node after a substitution (what `eval` does) gives the same value.
///
/// Oracle: mpmath at `x = 10³, 10⁶` with 30 and 60 digits (the values on
/// the negative axis are those from above, the principal branches `evalf`
/// uses): `besseli(1, -x)` = −2.4844·10⁴³², −1.2101·10⁴³⁴²⁹¹; `besseli(1/2,
/// -x)` = 2.4854·10⁴³²·i; `besseli(3/2, -x)`, `besseli(-1/2, -x)` negative
/// imaginary; `besseli(1/3, -x)` = (1.2428 + 2.1525i)·10⁴³² (direction
/// `e^{iπ/3}`); `besselk(ν, -x)` = −7.8·10⁴³²·i for ν = 0, 1, 1/2;
/// `expint(1, -x)` = −1.972·10⁴³¹ − 3.14159i; `expint(2, -x)` = −1.974·10⁴³¹
/// − 3141.59i (imaginary part −πx, unbounded); `expint(0, -x)`,
/// `expint(-1, -x)`, `expint(1/2, -x)` real negative; `gammainc(s, -x)`:
/// `s = 1` +1.97·10⁴³⁴, `s = 2` −1.97·10⁴³⁷, `s = 3` +1.97·10⁴⁴⁰, `s = 0`
/// −1.97·10⁴³¹ − 3.14159i, `s = −1` 1.97·10⁴²⁸ + 3.14159i, `s = 1/2`
/// −6.23·10⁴³²·i with real part `√π` (`Γ(1/2, −x) = √π − i√π·erfi(√x)`),
/// `s = 3/2`, `s = −1/2` positive imaginary; `ellipe(x)` = 0.000785 +
/// 999.996i at 10⁶ (real part → 0, imaginary side of the cut not chosen
/// by `evalf`: `zoo`), `ellipe(-x)` = 1000.0044; `chi(-x)` =
/// 1.5166·10⁴³⁴²⁸⁸ + 3.14159i.  SymPy 1.14 agrees on `besseli(1, -oo)` = -oo,
/// `besseli(1/2, -oo)` = oo*I, `uppergamma(2, -oo)` = -oo,
/// `lowergamma(2, -oo)` = oo, `lowergamma(1/2, -oo)` = oo*I,
/// `elliptic_e(-oo)` = oo; it gives `uppergamma(3, -oo)` = nan,
/// `uppergamma(-1, -oo)` = 0 (wrong) and `Chi(-oo)` = oo (no `iπ`).
#[test]
fn library_functions_at_minus_infinity_fold_to_their_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let neg_inf = ctx.neg_infinity();
    for (src, value) in [
        ("besseli(1, -oo)", "-oo"),
        ("besseli(0, -oo)", "oo"),
        ("besseli(2, -oo)", "oo"),
        ("besseli(1/2, -oo)", "I*oo"),
        ("besseli(3/2, -oo)", "-I*oo"),
        ("besseli(-1/2, -oo)", "-I*oo"),
        ("besseli(1/3, -oo)", "zoo"),
        ("besselk(0, -oo)", "-I*oo"),
        ("besselk(1, -oo)", "-I*oo"),
        ("besselk(1/2, -oo)", "-I*oo"),
        ("expint(1, -oo)", "-pi*I - oo"),
        ("expint(0, -oo)", "-oo"),
        ("expint(-1, -oo)", "-oo"),
        ("expint(1/2, -oo)", "-oo"),
        ("expint(2, -oo)", "zoo"),
        ("uppergamma(1, -oo)", "oo"),
        ("uppergamma(2, -oo)", "-oo"),
        ("uppergamma(3, -oo)", "oo"),
        ("uppergamma(0, -oo)", "-pi*I - oo"),
        ("uppergamma(-1, -oo)", "pi*I + oo"),
        ("uppergamma(1/2, -oo)", "sqrt(pi) - I*oo"),
        ("uppergamma(3/2, -oo)", "1/2*sqrt(pi) + I*oo"),
        ("uppergamma(-1/2, -oo)", "-2*sqrt(pi) + I*oo"),
        ("uppergamma(1/3, -oo)", "zoo"),
        ("lowergamma(2, -oo)", "oo"),
        ("lowergamma(1/2, -oo)", "I*oo"),
        ("elliptic_e(oo)", "zoo"),
        ("elliptic_e(-oo)", "oo"),
        ("Chi(-oo)", "pi*I + oo"),
    ] {
        let e = parse(&ctx, src);
        assert_eq!(e.to_string(), value, "{src} when built");
        assert_eq!(e.eval().to_string(), value, "{src} evaluated");
        // The same function at `x` (`−x` for `+∞`), then `x = −∞`.
        let at_x = parse(&ctx, &src.replace("-oo", "x").replace("(oo)", "(-x)"));
        let substituted = at_x.subs(&x, &neg_inf);
        assert_eq!(substituted.eval().to_string(), value, "{src} after subs");
    }
}

/// `0·f(±∞)` is not 0 when `f` is infinite there: `0·∞` is `nan`.  Up to
/// 0.52 the seven products of the wave-7 hunt (F1 h3: 23 wrong of 520)
/// were 0.  `0/f(−∞)` with `f` infinite is `0`.
///
/// Oracle: the values of
/// [`library_functions_at_minus_infinity_fold_to_their_values`]; `0·∞` and
/// `0·zoo` are nan, `0/∞` is 0 (SymPy 1.14: `0*besseli(1, -oo)` → nan,
/// `0*uppergamma(2, -oo)` → nan, `0*elliptic_e(oo)` → nan, `0*Chi(-oo)` →
/// nan; it gives 0 for `0*besselk(1, -oo)` and `0*expint(1, -oo)`, which
/// it leaves unevaluated).
#[test]
fn zero_times_a_function_infinite_at_infinity_is_nan() {
    let ctx = Context::new();
    for src in [
        "0*besseli(1, -oo)",
        "0*besselk(1, -oo)",
        "0*expint(1, -oo)",
        "0*uppergamma(2, -oo)",
        "0*elliptic_e(oo)",
        "0*elliptic_e(-oo)",
        "0*Chi(-oo)",
        "0*uppergamma(1/2, -oo)",
        "0*lowergamma(1/2, -oo)",
    ] {
        let e = parse(&ctx, src);
        assert_eq!(e.to_string(), "nan", "{src}");
        assert_ne!(e.is_zero(), Some(true), "{src}");
    }
    assert_eq!(parse(&ctx, "0/besseli(1, -oo)").to_string(), "0");
    assert_eq!(parse(&ctx, "0/uppergamma(3, -oo)").to_string(), "0");
}

/// A library function whose values at `±∞` fold when built, left
/// unevaluated at `±∞`, has no value the constructors know, so `0·f` and
/// `0/f` are not folded to 0 (D11): `I_ν(−∞)` for a symbolic order is
/// infinite (`|I_ν(−x)| = |e^{iπν}|·I_ν(x) → ∞`), `Ai′(−∞)` oscillates with
/// growing amplitude.  Up to 0.52 both were 0.  The values `eval` knows at
/// infinity fold when built, so their products stay decided:
/// `0·fresnels(∞) = 0` (`S(∞) = 1/2`), `0·lowergamma(2, ∞) = 0` (`γ(2, ∞) =
/// Γ(2) = 1`), `0·dirichlet_eta(∞) = 0` (`η(∞) = 1`); functions whose values
/// at infinity are not folded keep their reading (`0·atan(∞) = 0`,
/// `x·chebyshev_u(0, ∞)` vanishes at `x = 0`: `U₀ = 1`).
///
/// Oracle: DLMF 10.40.1 (`I_ν(x) ~ e^x/√(2πx)` for every ν), 9.7.10
/// (`Ai′(−x) ~ x^{1/4}·cos(⅔x^{3/2} + π/4)/√π`); SymPy 1.14:
/// `0*besseli(nu, -oo)` → nan, `fresnels(oo)` → 1/2, `dirichlet_eta(oo)` →
/// 1 (it gives `0*airyaiprime(-oo)` → 0 and `lowergamma(2, oo)` → nan).
#[test]
fn zero_times_a_library_function_without_a_value_at_infinity_is_not_zero() {
    let ctx = Context::new();
    for src in [
        "0*besseli(nu, -oo)",
        "0*airyaiprime(-oo)",
        "0/airyaiprime(-oo)",
        "0*dirichlet_eta(-oo)",
    ] {
        assert_eq!(parse(&ctx, src).to_string(), "nan", "{src}");
    }
    for (src, value) in [
        ("fresnels(oo)", "1/2"),
        ("fresnelc(-oo)", "-1/2"),
        ("dirichlet_eta(oo)", "1"),
    ] {
        assert_eq!(parse(&ctx, src).to_string(), value, "{src}");
    }
    assert_eq!(parse(&ctx, "lowergamma(2, oo)").eval().to_string(), "1");
    for src in [
        "0*fresnels(oo)",
        "0*lowergamma(2, oo)",
        "0*dirichlet_eta(oo)",
        "0*besselj(1, oo)",
        "0*atan(oo)",
        "0*Si(oo)",
        "0*chebyshev_u(0, oo)",
    ] {
        assert_eq!(parse(&ctx, src).to_string(), "0", "{src}");
    }
}

/// `δ` and `H` have values only at real arguments, so `0·δ(c)` and
/// `0·H(c)` at a constant `c` proved non-real are `nan` (D11), not 0.  Up
/// to 0.52 `0·DiracDelta(I)`, `0·Heaviside(1 + I)` and `0/DiracDelta(2I)`
/// were 0, while `DiracDelta(I)` itself had no value (`eval_decimal` an
/// error, `is_zero` None since 0.51).  At a real or symbolic argument the
/// product is 0 as before.
///
/// Oracle: SymPy 1.14 `DiracDelta(I)` and `Heaviside(I)` raise
/// `ValueError("Function defined only for Real Values")`; `δ(√2 − 1) = 0`
/// (`√2 − 1 ≠ 0` real).
#[test]
fn delta_and_heaviside_off_the_reals_times_zero_is_nan() {
    let ctx = Context::new();
    for src in [
        "0*DiracDelta(I)",
        "0*Heaviside(1 + I)",
        "0*Heaviside(exp(I))",
        "0*DiracDelta(sqrt(-2))",
        "0/DiracDelta(2*I)",
    ] {
        assert_eq!(parse(&ctx, src).to_string(), "nan", "{src}");
    }
    for src in [
        "0*DiracDelta(x)",
        "0*Heaviside(x)",
        "0*DiracDelta(sqrt(2) - 1)",
    ] {
        assert_eq!(parse(&ctx, src).to_string(), "0", "{src}");
    }
}

/// `cancel` of a quotient whose denominator vanishes identically by an
/// identity of its functions is `nan`, as `ratsimp`, `simplify` and
/// `expand` give.  `cancel` reads `sin x` and `cos x` as independent
/// indeterminates: up to 0.52 it returned the cancelled numerator, `0` for
/// `((x + 1)² − x² − 2x − 1)/((sin²x + cos²x − 1)(x + 2))` and `y` with
/// `+ y` (wave-6 sweep, B7 seeds zero 136, 217, zero2 124, 613).  A
/// denominator proved not identically zero keeps the cancelled result.
///
/// Oracle: `sin²x + cos²x = 1`, so the quotient is `0/0` at every point;
/// `sin x − 1` is −1 at `x = 0`.  SymPy 1.14 `cancel` gives `0` and `y`
/// (and `ratsimp(N0/D + y)` → `y`): deliberately different.
#[test]
fn cancel_of_a_quotient_with_an_identically_zero_denominator_is_nan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let q = ctx.symbol_with("q", &[Assumption::Negative]).unwrap();
    let _ = q;
    for src in [
        "((x + 1)**2 - x**2 - 2*x - 1)/((sin(x)**2 + cos(x)**2 - 1)*(x + 2))",
        "((x + 1)**2 - x**2 - 2*x - 1)/(sin(x)**2 + cos(x)**2 - 1) + y",
        "((x + 1)**2 - x**2 - 2*x - 1)/((sin(x)**2 + cos(x)**2 - 1)*(x + 2)) + y",
        "((x + 1)**2 - x**2 - 2*x - 1)/(sin(x)**2 + cos(x)**2 - 1 + Heaviside(q))",
    ] {
        let e = parse(&ctx, src);
        assert_eq!(e.cancel(&x).to_string(), "nan", "cancel {src}");
        assert_eq!(e.ratsimp().to_string(), "nan", "ratsimp {src}");
    }
    for (src, value) in [
        ("((x + 1)**2 - x**2 - 2*x - 1)/(x - 1)", "0"),
        ("((x + 1)**2 - x**2 - 2*x - 1)/(sin(x) - 1)", "0"),
        ("(sin(x)**2 - 1)/(sin(x) - 1)", "sin(x) + 1"),
        ("(x**2 - 1)/(x - 1)", "x + 1"),
    ] {
        assert_eq!(parse(&ctx, src).cancel(&x).to_string(), value, "{src}");
    }
}

/// `exp(c·ln p) = p^c` and, for a positive `p`, `ln(p^c) = c·ln p` are seen
/// by the zero tests of the canonical constructors: a denominator that
/// vanishes by them is `0`, so `0/d` and `d/d` are `nan`.  Up to 0.52 the
/// numerator `⌈n⌉ − n` (0 for an integer `n`, folded when built) over
/// `exp(ln p) − p + cosh²x − sinh²x − 1` was `0` with `is_zero() =
/// Some(true)` (wave-6 sweep, B6 zero class), `(exp(ln p) − p)·(y + 1)/((exp(ln
/// p) − p)·(x + 2))` was `(y + 1)/(x + 2)`, and `sin(πn)/((ln p² − 2 ln
/// p)(y + 1))` for a composite `n` was `0`.  A trigonometric identity in
/// both parts of a quotient cancels no more: `(sin²x + cos²x − 1)·(y +
/// 1)/(sin²x + cos²x − 1)` was `y + 1`.
///
/// Oracle: exact identities `e^{Log p} = p` (every `p ≠ 0`), `Log(p²) =
/// 2·Log p` (`p > 0`), `cosh² − sinh² = 1`, `sin² + cos² = 1`;
/// `0/0` is nan.  SymPy 1.14 gives `0` for the first quotient.
#[test]
fn zero_over_a_denominator_vanishing_by_log_identities_is_nan() {
    let ctx = Context::new();
    let _n = ctx.symbol_with("n", &[Assumption::Integer]).unwrap();
    let _p = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    let _m = ctx.symbol_with("m", &[Assumption::Composite]).unwrap();
    for src in [
        "(ceiling(n) - n)/(exp(log(p)) - p + cosh(x)**2 - sinh(x)**2 - 1)",
        "(ceiling(n) - n)/(exp(log(y)) - y + cosh(x)**2 - sinh(x)**2 - 1)",
        "(exp(log(p)) - p)*(y + 1)/((exp(log(p)) - p)*(x + 2))",
        "sin(pi*m)/((log(p**2) - 2*log(p))*(y + 1))",
        "(sin(x)**2 + cos(x)**2 - 1)*(y + 1)/(sin(x)**2 + cos(x)**2 - 1)",
        "(tan(x)*cos(x) - sin(x))/((tan(x)*cos(x) - sin(x))*(x + 2))",
    ] {
        let e = parse(&ctx, src);
        assert_eq!(e.to_string(), "nan", "{src}");
        assert_ne!(e.is_zero(), Some(true), "{src}");
    }
    // Not identically zero: unchanged.
    assert_eq!(parse(&ctx, "(sin(x) + 1)/(sin(x) + 1)").to_string(), "1");
    assert_eq!(
        parse(&ctx, "(x + sin(x))*(y + 1)/(x + sin(x))").to_string(),
        "y + 1"
    );
    // `ln(y²) − 2·ln y` is not 0 for a complex `y` (`2πi` for `y = −1`).
    assert_ne!(
        parse(&ctx, "1/(log(y**2) - 2*log(y))")
            .simplify()
            .to_string(),
        "zoo"
    );
    assert_eq!(
        parse(&ctx, "1/(exp(log(p)) - p)").ratsimp().to_string(),
        "zoo"
    );
}
