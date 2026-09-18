//! symplex 0.2 base-layer fixes — display precedence for `Pow` bases.
//!
//! A rational or negative numeric base must be parenthesised:
//! `(2/3)^(-1/2)`, not `2/3^(-1/2)` (which re-parses as `2/(3^(-1/2))`).
//! Every case is checked as a display → parse → display round trip and
//! the LaTeX form is checked for `\left( … \right)`.

use symplex::prelude::*;

fn s<T: std::fmt::Display>(e: &T) -> String {
    format!("{e}")
}

fn roundtrip(ctx: &Context, e: &Ex) {
    let text = s(e);
    let back = ctx
        .parse(&text)
        .unwrap_or_else(|err| panic!("failed to parse `{text}`: {err}"));
    assert_eq!(
        back, *e,
        "display → parse → display changed `{text}` into `{back}`"
    );
    assert_eq!(s(&back), text);
}

#[test]
fn rational_base_negative_fractional_exponent() {
    let ctx = Context::new();
    // Numeric radicals are rationalised at construction (0.2 numfix):
    // (2/3)^(-1/2) = √(3/2) = √6/2.  The parenthesised `(p/q)^(…)` display
    // is exercised with a symbolic exponent below.
    let e = ctx.rational(2, 3).pow(&ctx.rational(-1, 2));
    assert_eq!(s(&e), "1/2*sqrt(6)");
    roundtrip(&ctx, &e);
    let x = ctx.symbol("x");
    let f = ctx.rational(2, 3).pow(&(-&x / 2));
    assert_eq!(s(&f), "(2/3)^(-1/2*x)");
    roundtrip(&ctx, &f);
}

#[test]
fn rational_base_symbolic_exponent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.rational(2, 3).pow(&x);
    assert_eq!(s(&e), "(2/3)^x");
    roundtrip(&ctx, &e);
    let f = ctx.rational(1, 2).pow(&(&x + 1));
    assert_eq!(s(&f), "(1/2)^(x + 1)");
    roundtrip(&ctx, &f);
}

#[test]
fn negative_integer_base_symbolic_exponent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.int(-2).pow(&x);
    assert_eq!(s(&e), "(-2)^x");
    roundtrip(&ctx, &e);
    let n = ctx.symbol("n");
    let f = ctx.int(-1).pow(&n);
    assert_eq!(s(&f), "(-1)^n");
    roundtrip(&ctx, &f);
}

#[test]
fn negative_rational_base() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.rational(-3, 4).pow(&x);
    assert_eq!(s(&e), "(-3/4)^x");
    roundtrip(&ctx, &e);
}

#[test]
fn rational_base_inside_products_and_sums() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = ctx.rational(2, 3).pow(&x);
    let e = &x * &p + 1;
    assert_eq!(s(&e), "x*(2/3)^x + 1");
    roundtrip(&ctx, &e);
    let f = &(-&x) * &p;
    assert_eq!(s(&f), "-x*(2/3)^x");
    roundtrip(&ctx, &f);
}

#[test]
fn rational_base_radical_forms_rationalised() {
    let ctx = Context::new();
    // sqrt / cbrt of a rational are rationalised (0.2 numfix normal form):
    // √(2/3) = √6/3, ∛(2/3) = ∛2·3^(2/3)/3 (the same form SymPy produces).
    let e = ctx.rational(2, 3).sqrt();
    assert_eq!(s(&e), "1/3*sqrt(6)");
    roundtrip(&ctx, &e);
    let c = ctx.rational(2, 3).pow(&ctx.rational(1, 3));
    assert_eq!(s(&c), "1/3*cbrt(2)*3^(2/3)");
    roundtrip(&ctx, &c);
}

#[test]
fn positive_integer_bases_need_no_parens() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(s(&ctx.int(2).pow(&x)), "2^x");
    assert_eq!(s(&ctx.int(2).pow(&ctx.rational(3, 2))), "2^(3/2)");
    assert_eq!(s(&x.pow(&ctx.rational(2, 3))), "x^(2/3)");
    roundtrip(&ctx, &ctx.int(2).pow(&x));
    roundtrip(&ctx, &ctx.int(2).pow(&ctx.rational(3, 2)));
}

#[test]
fn latex_parenthesises_rational_and_negative_bases() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.rational(2, 3).pow(&x);
    let l = e.to_latex();
    assert!(
        l.contains(r"\left(\frac{2}{3}\right)^{x}"),
        "rational base needs parens in LaTeX: {l}"
    );
    let f = ctx.int(-2).pow(&x);
    let l = f.to_latex();
    assert!(
        l.contains(r"\left(-2\right)^{x}"),
        "negative base needs parens in LaTeX: {l}"
    );
    let g = ctx.int(2).pow(&x);
    assert_eq!(g.to_latex(), "2^{x}");
}

#[test]
fn latex_nested_pow_avoids_double_superscript() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    // (x^y)^z stays nested (non-integer exponents) and must not render
    // as `x^{y}^{z}`, which is a LaTeX "double superscript" error.
    let e = x.pow(&y).pow(&z);
    let l = e.to_latex();
    assert!(!l.contains("}^{"), "double superscript: {l}");
    assert!(l.contains(r"\left(x^{y}\right)^{z}"), "{l}");
}
