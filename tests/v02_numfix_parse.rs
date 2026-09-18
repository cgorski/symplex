//! Regression tests for parser coverage (0.2 numfix): `binomial`, `cot`/
//! `sec`/`csc`, `besselj`/`bessely`/`besseli`/`besselk`, `beta`, `atan2`,
//! `min`/`max` (variadic), `polygamma`, factorial (`factorial(n)` and
//! postfix `n!`), `Sum`/`Product` (both the 4-argument SymPy form and the
//! displayed `k=lo..hi` form).  Every construction is checked as a
//! display → parse → display round trip.

use symplex::prelude::*;

fn roundtrip(ctx: &Context, e: &Ex) {
    let text = format!("{e}");
    let back = ctx
        .parse(&text)
        .unwrap_or_else(|err| panic!("failed to parse `{text}`: {err}"));
    assert_eq!(back, *e, "display → parse changed `{text}` into `{back}`");
    assert_eq!(format!("{back}"), text);
}

#[test]
fn parse_binomial_round_trip() {
    let ctx = Context::new();
    let (n, k) = (ctx.symbol("n"), ctx.symbol("k"));
    roundtrip(&ctx, &n.binomial(&k));
    assert_eq!(ctx.parse("binomial(n, k)").unwrap(), n.binomial(&k));
    assert_eq!(ctx.parse("binomial(5, 2)").unwrap().eval(), ctx.int(10));
}

#[test]
fn parse_reciprocal_trig_round_trip() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (name, e) in [
        ("cot", x.cot()),
        ("sec", x.sec()),
        ("csc", x.csc()),
        ("coth", x.coth()),
        ("sech", x.sech()),
        ("csch", x.csch()),
    ] {
        roundtrip(&ctx, &e);
        assert_eq!(ctx.parse(&format!("{name}(x)")).unwrap(), e, "{name}");
    }
    assert_eq!(ctx.parse("cot(pi/4)").unwrap().eval(), ctx.int(1));
}

#[test]
fn parse_bessel_round_trip() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let n = ctx.symbol("n");
    for e in [
        x.bessel_j(&ctx.int(0)),
        x.bessel_y(&n),
        x.bessel_i(&ctx.int(1)),
        x.bessel_k(&ctx.rational(1, 2)),
    ] {
        roundtrip(&ctx, &e);
    }
    // Order first, as in SymPy: besselj(0, 20) = J0(20).
    let j = ctx.parse("besselj(0, 20)").unwrap();
    assert_eq!(j, ctx.int(20).bessel_j(&ctx.int(0)));
    assert!((j.eval_f64().unwrap() - 0.16702466434058315).abs() < 1e-14);
}

#[test]
fn parse_beta_and_atan2_round_trip() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    roundtrip(&ctx, &x.beta(&y));
    assert_eq!(ctx.parse("beta(x, y)").unwrap(), x.beta(&y));
    roundtrip(&ctx, &y.atan2(&x));
    assert_eq!(ctx.parse("atan2(y, x)").unwrap(), y.atan2(&x));
}

#[test]
fn parse_min_max_variadic_round_trip() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    roundtrip(&ctx, &x.min_with(&y));
    roundtrip(&ctx, &Ex::max_of(&ctx, [x.clone(), y.clone(), z.clone()]));
    assert_eq!(ctx.parse("min(1, 2, 3, 4, 5)").unwrap().eval(), ctx.int(1));
    assert_eq!(ctx.parse("max(1, 2, 3, 4, 5)").unwrap().eval(), ctx.int(5));
    assert!(ctx.parse("min(1)").is_err(), "min needs ≥ 2 arguments");
}

#[test]
fn parse_polygamma_round_trip() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    roundtrip(&ctx, &x.polygamma(&ctx.int(2)));
    assert_eq!(
        ctx.parse("polygamma(2, x)").unwrap(),
        x.polygamma(&ctx.int(2))
    );
}

#[test]
fn parse_factorial_function_and_postfix() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    roundtrip(&ctx, &n.factorial());
    roundtrip(&ctx, &(&n + 1).factorial());
    assert_eq!(ctx.parse("factorial(n)").unwrap(), n.factorial());
    assert_eq!(ctx.parse("n!").unwrap(), n.factorial());
    assert_eq!(ctx.parse("(n + 1)!").unwrap(), (&n + 1).factorial());
    assert_eq!(ctx.parse("5!").unwrap().eval(), ctx.int(120));
    // Postfix binds tighter than every infix operator.
    assert_eq!(ctx.parse("2^3!").unwrap().eval(), ctx.int(64));
    assert_eq!(ctx.parse("3!^2").unwrap().eval(), ctx.int(36));
    assert_eq!(ctx.parse("2*3!").unwrap().eval(), ctx.int(12));
}

#[test]
fn parse_sum_and_product_round_trip() {
    let ctx = Context::new();
    let (n, k) = (ctx.symbol("n"), ctx.symbol("k"));
    let sum = Ex::symbolic_sum(&k.powi(2), &k, &ctx.int(1), &n);
    let prod = Ex::symbolic_product(&(&k + 1), &k, &ctx.int(1), &n);
    // Displayed `Sum(k^2, k=1..n)` form …
    roundtrip(&ctx, &sum);
    roundtrip(&ctx, &prod);
    // … and the 4-argument SymPy form.
    assert_eq!(ctx.parse("Sum(k^2, k, 1, n)").unwrap(), sum);
    assert_eq!(ctx.parse("Product(k + 1, k, 1, n)").unwrap(), prod);
    assert_eq!(ctx.parse("Sum(k, k, 1, 10)").unwrap().eval(), ctx.int(55));
    // The index must be a symbol.
    assert!(ctx.parse("Sum(k^2, 2, 1, n)").is_err());
    assert!(ctx.parse("Sum(k^2, k+1=1..n)").is_err());
}

#[test]
fn parse_lambertw_display_alias() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    roundtrip(&ctx, &x.lambertw());
    assert_eq!(ctx.parse("lambertw(x)").unwrap(), x.lambertw());
}

#[test]
fn parse_unknown_function_still_errors() {
    let ctx = Context::new();
    let err = ctx.parse("frobnicate(x)").err().expect("unknown function");
    assert!(err.to_string().contains("unknown function"), "{err}");
    let err = ctx.parse("besselj(x)").err().expect("wrong arity");
    assert!(err.to_string().contains("unknown function"), "{err}");
    assert!(ctx.parse("1 = 2").is_err());
    assert!(ctx.parse("1..2").is_err());
}
