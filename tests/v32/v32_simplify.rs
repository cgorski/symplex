//! `simplify` of expressions whose denominator vanishes only by an
//! identity of its functions: undefined everywhere, so `nan` (`0/0`) or
//! `zoo` (`P/0`), as for denominators that vanish as polynomials since
//! 0.32.

use symplex::prelude::*;

fn simplified(ctx: &Context, src: &str) -> String {
    ctx.parse(src).unwrap().simplify().to_string()
}

/// `((x + 1)² − x² − 2x − 1)/(tan x·cos x − sin x)` simplified to `0`:
/// the strategies multiply the numerator out to 0, and the 0.32 test for a
/// vanishing denominator saw `tan x`, `cos x`, `sin x` as independent
/// indeterminates.  The denominator is 0 for every `x`, so the expression
/// is `0/0`: `nan`.  So is `(x + 1)·(sin 2x − 2·sin x·cos x)/(cosh²x −
/// sinh²x − 1)`, which was `zoo` (the numerator vanishes by the double
/// angle formula, not once multiplied out).
///
/// SymPy 1.14 (we deliberately differ, as for `together`/`ratsimp`):
/// `simplify(((x+1)**2-x**2-2*x-1)/(tan(x)*cos(x)-sin(x)))` → `0`;
/// `simplify((x+1)*(sin(2*x)-2*sin(x)*cos(x))/(cosh(x)**2-sinh(x)**2-1))`
/// → `0`.
#[test]
fn denominators_zero_by_an_identity_are_nan() {
    let ctx = Context::new();
    for src in [
        "((x+1)^2-x^2-2*x-1)/(tan(x)*cos(x)-sin(x))",
        "(x+1)*(sin(2*x)-2*sin(x)*cos(x))/(cosh(x)^2-sinh(x)^2-1)",
        "(sin(x)^2+cos(x)^2-1)/(tan(x)*cos(x)-sin(x))",
        "(exp(x+y)-exp(x)*exp(y))/(sin(x+y)-sin(x)*cos(y)-cos(x)*sin(y))",
        "((x+1)^2-x^2-2*x-1)/sqrt(tan(x)*cos(x)-sin(x))",
        "(sin(x)^2+cos(x)^2-1)/(cosh(x)^2-sinh(x)^2-1) + x",
    ] {
        assert_eq!(simplified(&ctx, src), "nan", "{src}");
    }
}

/// A numerator that vanishes only by an identity over a denominator that
/// vanishes as a polynomial: `zoo` before, `nan` now.
///
/// SymPy: `simplify((sin(2*x)-2*sin(x)*cos(x))/((x+1)**2-x**2-2*x-1))` →
/// `zoo` (SymPy builds `1/0 = zoo` first and does not look at the
/// numerator).
#[test]
fn numerators_zero_by_an_identity_are_nan() {
    let ctx = Context::new();
    let src = "(sin(2*x)-2*sin(x)*cos(x))/((x+1)^2-x^2-2*x-1)";
    assert_eq!(simplified(&ctx, src), "nan");
}

/// What did not change: a nonzero numerator over a vanishing denominator
/// is `zoo`, a zero absorbed by a further division stays `0` (the 0.32
/// arithmetic of `1/0 = zoo`), and `ln(x²) − 2·ln x` is not 0 over ℂ
/// (`−2πi` for `Re x < 0`), so `1/(ln(x²) − 2·ln x)` is left alone — but it
/// is 0 for a positive `p`.
///
/// SymPy: `simplify(1/(sin(x)**2+cos(x)**2-1))` → `zoo`;
/// `simplify(x/(tan(x)*cos(x)-sin(x)))` → `zoo*x`;
/// `simplify(x/(1+1/(tan(x)*cos(x)-sin(x))))` → `0`;
/// `simplify(1/(log(x**2)-2*log(x)))` → `-1/(2*log(x) - log(x**2))`;
/// `simplify(1/(log(p**2)-2*log(p)))` with `p` positive → `zoo`.
#[test]
fn other_vanishing_denominators_unchanged() {
    let ctx = Context::new();
    let _ = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    for (src, want) in [
        ("1/(sin(x)^2+cos(x)^2-1)", "zoo"),
        ("x/(tan(x)*cos(x)-sin(x))", "zoo"),
        ("x + 1/(tan(x)*cos(x)-sin(x))", "zoo"),
        ("x/(1+1/(tan(x)*cos(x)-sin(x)))", "0"),
        ("1/(log(p^2)-2*log(p))", "zoo"),
        ("1/((1+sqrt(2))^2-3-2*sqrt(2))", "zoo"),
    ] {
        assert_eq!(simplified(&ctx, src), want, "{src}");
    }
    let e = ctx.parse("1/(log(x^2)-2*log(x))").unwrap();
    let s = e.simplify();
    assert!(s.to_string().contains("ln(x^2)"), "{s}");
}

/// Correct simplifications are untouched by the new test (it runs only on
/// denominators with a function in them, and certifies a nonzero value at
/// the first sample point).
///
/// SymPy: `simplify((x**2-1)/(x-1))` → `x + 1`; `simplify(sin(x)/x)` →
/// `sin(x)/x`; `simplify((sin(x)**2+cos(x)**2)/(x+1))` → `1/(x + 1)`.
#[test]
fn ordinary_simplifications_unchanged() {
    let ctx = Context::new();
    for (src, want) in [
        ("(x^2-1)/(x-1)", "x + 1"),
        ("sin(x)/x", "sin(x)/x"),
        ("(sin(x)^2+cos(x)^2)/(x+1)", "1/(x + 1)"),
    ] {
        assert_eq!(simplified(&ctx, src), want, "{src}");
    }
}
