//! Rational functions and factoring: needless refusals removed in 0.43.
//! Each test says what was returned before (the input unchanged, unless
//! stated) and cites its oracle: SymPy 1.14 (`cancel`, `apart`, `factor`,
//! `factor(…, extension=…)`), and an exact identity (the result minus the
//! input is 0 as a rational function, by `ratsimp`).

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `a − b` is 0 as a rational function of its generators.
fn assert_same_function(a: &Ex, b: &Ex) {
    let d = (a - b).ratsimp();
    assert!(d.is_zero_structural(), "{a} - ({b}) = {d}");
}

/// Number of terms of a sum (1 for anything else).
fn terms(e: &Ex) -> usize {
    let s = e.to_string();
    let mut depth = 0i32;
    let mut n = 1;
    let b = s.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        match c {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b'+' | b'-' if depth == 0 && i > 0 && b[i - 1] == b' ' => n += 1,
            _ => {}
        }
    }
    n
}

// ── Sums of fractions ─────────────────────────────────────────────────

/// Was unchanged: `apart` split only a single quotient, and a sum looked
/// like a fraction with denominator 1.  SymPy 1.14:
/// `apart(1/(x - 1) + 1/((x - 1)*(x + 1)), x)` = `-1/(2*(x + 1)) + 3/(2*(x - 1))`.
#[test]
fn partial_fractions_of_a_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "1/(x - 1) + 1/((x - 1)*(x + 1))");
    let r = e.partial_fractions(&x);
    assert_eq!(r.to_string(), "-1/(2*(x + 1)) + 3/(2*(x - 1))");
    assert_same_function(&r, &e);
}

/// Was unchanged.  SymPy 1.14: `cancel(1/(x - 1) + 1/((x - 1)*(x + 1)))` =
/// `(x + 2)/(x**2 - 1)`; `cancel(1/a + 1/(a*x), x)` = `(x + 1)/(a*x)`.
#[test]
fn cancel_combines_a_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "1/(x - 1) + 1/((x - 1)*(x + 1))");
    assert_eq!(e.cancel(&x).to_string(), "(x + 2)/(x^2 - 1)");
    let e = parse(&ctx, "1/a + 1/(a*x)");
    let r = e.cancel(&x);
    assert_eq!(r.to_string(), "(x + 1)/(a*x)");
    assert_same_function(&r, &e);
}

/// Unchanged before and after: a sum that is already split, and a single
/// quotient with nothing to cancel keeps its factored form.  SymPy 1.14:
/// `apart(x + 1/(x + 1), x)` = `x + 1/(x + 1)`.
#[test]
fn nothing_to_do_stays_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "x + 1/(x + 1)");
    assert_eq!(e.partial_fractions(&x), e);
    let e = parse(&ctx, "(x + 1)*(x + 2)/(x + 3)");
    assert_eq!(e.cancel(&x), e);
    let e = parse(&ctx, "(x + a)/(x + b)");
    assert_eq!(e.cancel(&x), e);
}

// ── Symbolic parameters ───────────────────────────────────────────────

/// Was unchanged (the conversion required rational coefficients).  SymPy
/// 1.14: `cancel((x**2 - a**2)/(x - a), x)` = `a + x`,
/// `cancel((x**2/a - a)/(x - a), x)` = `(a + x)/a`,
/// `cancel((a*x**2 - a)/(2*x - 2), x)` = `a*x/2 + a/2`.
#[test]
fn cancel_with_parameters() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (s, want) in [
        ("(x^2 - a^2)/(x - a)", "a + x"),
        ("(x^2/a - a)/(x - a)", "(a + x)/a"),
        ("(a*x^2 - a)/(2*x - 2)", "a*x/2 + a/2"),
    ] {
        let e = parse(&ctx, s);
        let r = e.cancel(&x);
        assert_same_function(&r, &e);
        assert_same_function(&r, &parse(&ctx, want));
        assert!(!r.to_string().contains("x^2"), "{s}: {r}");
    }
}

/// Was unchanged.  SymPy 1.14: `cancel((x*sin(x) + x)/x)` = `sin(x) + 1`
/// (functions are independent indeterminates, as in `ratsimp`).
#[test]
fn cancel_treats_functions_as_indeterminates() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "(x*sin(x) + x)/x");
    assert_eq!(e.cancel(&x), parse(&ctx, "sin(x) + 1"));
}

/// Was unchanged.  SymPy 1.14:
/// `apart((a - 9)/((3*x + 4)*(2*x - 3)**2), x)` =
/// `9*(a - 9)/(289*(3*x + 4)) - 6*(a - 9)/(289*(2*x - 3)) + 2*(a - 9)/(17*(2*x - 3)**2)`.
#[test]
fn partial_fractions_with_a_parameter_in_the_numerator() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "(a - 9)/((3*x + 4)*(2*x - 3)^2)");
    let r = e.partial_fractions(&x);
    let sympy = parse(
        &ctx,
        "9*(a - 9)/(289*(3*x + 4)) - 6*(a - 9)/(289*(2*x - 3)) + 2*(a - 9)/(17*(2*x - 3)^2)",
    );
    assert_same_function(&r, &sympy);
    assert_eq!(terms(&r), 3, "{r}");
}

/// Was unchanged.  SymPy 1.14: `apart(1/((x - a)*(x - b)), x)` =
/// `-1/((a - b)*(-b + x)) + 1/((-a + x)*(a - b))`;
/// `apart(x/((x - a)**2*(x + 1)), x)` =
/// `a/((-a + x)**2*(a + 1)) - 1/((a + 1)**2*(x + 1)) + 1/((-a + x)*(a + 1)**2)`;
/// `apart((x**2 + a)/(x - 1), x)` = `x + (a + 1)/(x - 1) + 1`.
#[test]
fn partial_fractions_with_parameters_in_the_denominator() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "1/((x - a)*(x - b))");
    let r = e.partial_fractions(&x);
    assert_eq!(
        r.to_string(),
        "-1/((a - b)*(-b + x)) + 1/((a - b)*(-a + x))"
    );
    let e = parse(&ctx, "x/((x - a)^2*(x + 1))");
    let r = e.partial_fractions(&x);
    let sympy = parse(
        &ctx,
        "a/((-a + x)^2*(a + 1)) - 1/((a + 1)^2*(x + 1)) + 1/((-a + x)*(a + 1)^2)",
    );
    assert_same_function(&r, &sympy);
    assert_eq!(terms(&r), 3, "{r}");
    assert!(r.to_string().contains("(a + 1)^"), "{r}");
    let e = parse(&ctx, "(x^2 + a)/(x - 1)");
    let r = e.partial_fractions(&x);
    assert_same_function(&r, &parse(&ctx, "x + (a + 1)/(x - 1) + 1"));
}

// ── Many variables ────────────────────────────────────────────────────

/// Was unchanged: more than four symbols were never factored.  SymPy
/// 1.14: `factor(expand((a + b)*(c + d + f)))` = `(a + b)*(c + d + f)`;
/// `factor(expand((a*b - c*d + 1)*(a + f*g - h)*(b*c + d - 2)))` =
/// `(a + f*g - h)*(a*b - c*d + 1)*(b*c + d - 2)`.
#[test]
fn factor_all_with_more_than_four_symbols() {
    let ctx = Context::new();
    let e = parse(&ctx, "(a + b)*(c + d + f)").expand();
    let r = e.factor_all();
    assert_eq!(r, parse(&ctx, "(a + b)*(c + d + f)"));
    let e = parse(&ctx, "(a*b - c*d + 1)*(a + f*g - h)*(b*c + d - 2)").expand();
    let (content, factors) = e.factor_list_all();
    assert_eq!(content.to_string(), "1");
    assert_eq!(factors.len(), 3, "{factors:?}");
    assert_same_function(&e.factor_all(), &e);
}

// ── Algebraic coefficients ────────────────────────────────────────────

/// Was unchanged (no factoring over algebraic number fields).  SymPy
/// 1.14: `factor(x**2 - 2*sqrt(2)*x + 2, extension=sqrt(2))` =
/// `(x - sqrt(2))**2`; `factor(x**4 + 1, extension=sqrt(2))` =
/// `(x**2 - sqrt(2)*x + 1)*(x**2 + sqrt(2)*x + 1)`;
/// `factor(x**4 + 1, extension=I)` = `(x**2 - I)*(x**2 + I)`;
/// `factor(x**3 - 2, extension=2**(1/3))` =
/// `(x - 2**(1/3))*(x**2 + 2**(1/3)*x + 2**(2/3))`;
/// `factor(x**4 - 5*x**2 + 6, extension=[sqrt(2), sqrt(3)])` = four
/// linear factors.  (SymPy's plain `factor` leaves the first unchanged:
/// `√2` is a generator there, and so it is in `factor`.)
#[test]
fn factor_over_algebraic_extensions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s2 = ctx.int(2).sqrt();
    let f = parse(&ctx, "x^2 - 2*sqrt(2)*x + 2");
    assert_eq!(f.factor(&x), f);
    assert_eq!(f.factor_extension(&x, &[]).to_string(), "(x - sqrt(2))^2");
    let g = parse(&ctx, "x^4 + 1");
    let r = g.factor_extension(&x, std::slice::from_ref(&s2));
    assert_eq!(
        r,
        parse(&ctx, "(x^2 - sqrt(2)*x + 1)*(x^2 + sqrt(2)*x + 1)")
    );
    let r = g.factor_extension(&x, &[ctx.i_unit()]);
    assert_eq!(r, parse(&ctx, "(x^2 - I)*(x^2 + I)"));
    let h = parse(&ctx, "x^3 - 2");
    let r = h.factor_extension(&x, &[parse(&ctx, "2^(1/3)")]);
    assert_eq!(terms(&r), 1, "{r}");
    assert_same_function(&r.expand(), &h);
    assert_eq!(r.to_string(), "(x - cbrt(2))*(x^2 + x*cbrt(2) + 2^(2/3))");
    let k = parse(&ctx, "x^4 - 5*x^2 + 6");
    let r = k.factor_extension(&x, &[s2, ctx.int(3).sqrt()]);
    assert_eq!(r.to_string().matches("x ").count(), 4, "{r}");
    assert_same_function(&r.expand(), &k);
}

/// Was unchanged.  SymPy 1.14 (default domain `ZZ_I` when `I` is the only
/// constant): `factor(x**2 + 2*I*x - 1)` = `(x + I)**2`,
/// `factor(2*I*x + 2)` = `2*I*(x - I)`, `factor(x**3 + I*x**2 + x + I)` =
/// `(x - I)*(x + I)**2`, but `factor(I*x**2 + 9*I)` = `I*(x**2 + 9)` (the
/// common factor `I` is taken out first and `x² + 9` has rational
/// coefficients); with `√2` a generator,
/// `factor(sqrt(2)*x**2 + sqrt(2)*x)` = `sqrt(2)*x*(x + 1)`.
#[test]
fn factor_uses_sympy_default_domains() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        parse(&ctx, "x^2 + 2*I*x - 1").factor(&x).to_string(),
        "(x + I)^2"
    );
    let e = parse(&ctx, "2*I*x + 2");
    let r = e.factor(&x);
    assert_eq!(r, parse(&ctx, "2*I*(x - I)"));
    let e = parse(&ctx, "x^3 + I*x^2 + x + I");
    assert_eq!(e.factor(&x), parse(&ctx, "(x - I)*(x + I)^2"));
    let e = parse(&ctx, "I*x^2 + 9*I");
    assert_eq!(e.factor(&x), parse(&ctx, "I*(x^2 + 9)"));
    let e = parse(&ctx, "sqrt(2)*x^2 + sqrt(2)*x");
    assert_eq!(e.factor(&x), parse(&ctx, "sqrt(2)*x*(x + 1)"));
    // Irreducible over ℚ: unchanged.
    let e = parse(&ctx, "x^2 + 1");
    assert_eq!(e.factor(&x), e);
}

/// Was unchanged.  SymPy 1.14 (`apart` defaults to `extension=True`):
/// `apart(sqrt(2)/(x**2 - 2), x)` = `-1/(2*(x + sqrt(2))) + 1/(2*(x - sqrt(2)))`,
/// `apart(I/(x**2 + 1), x)` = `-1/(2*(x + I)) + 1/(2*(x - I))`,
/// `apart(1/((x - sqrt(2))*(x**2 - 2)), x)` =
/// `1/(8*(x + sqrt(2))) - 1/(8*(x - sqrt(2))) + sqrt(2)/(4*(x - sqrt(2))**2)`;
/// `apart(1/(x**2 - 2), x)` is unchanged (no algebraic number in it).
#[test]
fn partial_fractions_over_algebraic_extensions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "sqrt(2)/(x^2 - 2)");
    let r = e.partial_fractions(&x);
    assert_eq!(r, parse(&ctx, "-1/(2*(x + sqrt(2))) + 1/(2*(x - sqrt(2)))"));
    let e = parse(&ctx, "I/(x^2 + 1)");
    let r = e.partial_fractions(&x);
    assert_eq!(r, parse(&ctx, "-1/(2*(x + I)) + 1/(2*(x - I))"));
    let e = parse(&ctx, "1/((x - sqrt(2))*(x^2 - 2))");
    let r = e.partial_fractions(&x);
    let sympy = parse(
        &ctx,
        "1/(8*(x + sqrt(2))) - 1/(8*(x - sqrt(2))) + sqrt(2)/(4*(x - sqrt(2))^2)",
    );
    assert_eq!((&r - &sympy).expand().simplify().to_string(), "0", "{r}");
    assert_eq!(terms(&r), 3, "{r}");
    let e = parse(&ctx, "1/(x^2 - 2)");
    assert_eq!(e.partial_fractions(&x), e);
}

/// Was unchanged.  SymPy 1.14: `cancel((x**2 - 2)/(x - sqrt(2)),
/// extension=sqrt(2))` = `x + sqrt(2)`; `cancel((x**2 + 1)/(x - I))` =
/// `x + I`.
#[test]
fn cancel_over_algebraic_extensions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "(x^2 - 2)/(x - sqrt(2))");
    assert_eq!(e.cancel(&x).to_string(), "x + sqrt(2)");
    let e = parse(&ctx, "(x^2 + 1)/(x - I)");
    assert_eq!(e.cancel(&x).to_string(), "x + I");
    let e = parse(&ctx, "(x^3 - 2)/(x^2 + 2^(1/3)*x + 2^(2/3))");
    let r = e.cancel(&x);
    assert_eq!(r, parse(&ctx, "x - 2^(1/3)"));
}
