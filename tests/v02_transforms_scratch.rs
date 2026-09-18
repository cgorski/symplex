//! Scratch exploration (will be deleted).
use symplex::prelude::*;

#[test]
#[ignore]
fn explore_limits() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let ninf = ctx.neg_infinity();
    let a = ctx.symbol_with("a", &[Assumption::Positive]);
    let b = ctx.symbol_with("b", &[Assumption::Positive]);

    let cases: Vec<(&str, Ex, Ex)> = vec![
        ("sin x / x @0", &x.sin() / &x, zero.clone()),
        ("(1-cos x)/x^2 @0", (1 - &x.cos()) / x.powi(2), zero.clone()),
        (
            "(e^x-1-x)/x^2 @0",
            (&x.exp() - 1 - &x) / x.powi(2),
            zero.clone(),
        ),
        ("x^(1/x) @inf", x.pow(&(1 / &x)), inf.clone()),
        ("(1+1/x)^x @inf", (1 + 1 / &x).pow(&x), inf.clone()),
        (
            "(1+a/x)^(bx) @inf",
            (1 + &a / &x).pow(&(&b * &x)),
            inf.clone(),
        ),
        (
            "(1+2/x)^(3x) @inf",
            (1 + 2 / &x).pow(&(&x * 3)),
            inf.clone(),
        ),
        (
            "sqrt(x^2+x)-x @inf",
            (x.powi(2) + &x).sqrt() - &x,
            inf.clone(),
        ),
        ("ln x / x^(1/2) @inf", &x.ln() / &x.sqrt(), inf.clone()),
        ("x e^-x @inf", &x * (-&x).exp(), inf.clone()),
        ("x^5/e^x @inf", x.powi(5) / x.exp(), inf.clone()),
        (
            "(tan x - sin x)/x^3 @0",
            (&x.tan() - &x.sin()) / x.powi(3),
            zero.clone(),
        ),
        (
            "(2^x-3^x)/x @0",
            (ctx.int(2).pow(&x) - ctx.int(3).pow(&x)) / &x,
            zero.clone(),
        ),
        ("ln(1+x)/x @0", (1 + &x).ln() / &x, zero.clone()),
        (
            "(x-sin x)/x^3 @0",
            (&x - &x.sin()) / x.powi(3),
            zero.clone(),
        ),
        ("atan x @inf", x.atan(), inf.clone()),
        ("atan x @-inf", x.atan(), ninf.clone()),
        ("erf x @inf", x.erf(), inf.clone()),
        ("erf x @-inf", x.erf(), ninf.clone()),
        ("tanh x @inf", x.tanh(), inf.clone()),
        ("tanh x @-inf", x.tanh(), ninf.clone()),
        ("W(x)/ln x @inf", &x.lambertw() / &x.ln(), inf.clone()),
        ("e^x - e^(2x) @inf", x.exp() - (&x * 2).exp(), inf.clone()),
        (
            "(e^x + e^(-x))/e^x @inf",
            (x.exp() + (-&x).exp()) / x.exp(),
            inf.clone(),
        ),
        (
            "(x^2+1)/(x+1) @inf",
            (x.powi(2) + 1) / (&x + 1),
            inf.clone(),
        ),
        ("|x| @0", x.abs(), zero.clone()),
        ("|x|/x @0", &x.abs() / &x, zero.clone()),
        ("floor(x) @1/2", x.floor(), ctx.rational(1, 2)),
        ("max(x, 1) @0", x.max_with(&ctx.int(1)), zero.clone()),
        ("min(x, 1) @inf", x.min_with(&ctx.int(1)), inf.clone()),
        ("ln x @0", x.ln(), zero.clone()),
        ("1/x @0", 1 / &x, zero.clone()),
        ("e^(1/x) @0", (1 / &x).exp(), zero.clone()),
        ("x ln x @0", &x * &x.ln(), zero.clone()),
        ("x^x @0", x.pow(&x), zero.clone()),
        ("sin(1/x) @0", (1 / &x).sin(), zero.clone()),
        ("tan x @pi/2", x.tan(), ctx.pi() / 2),
        ("H(x) @0", x.heaviside(), zero.clone()),
        ("sign(x) @0", x.sign(), zero.clone()),
        ("1/(1+e^(1/x)) @0", 1 / (1 + (1 / &x).exp()), zero.clone()),
        ("x sin(1/x) @inf", &x * (1 / &x).sin(), inf.clone()),
        ("x sin(1/x) @0", &x * (1 / &x).sin(), zero.clone()),
        ("sin x @inf", x.sin(), inf.clone()),
        ("x sin x @inf", &x * x.sin(), inf.clone()),
        ("ln(x)/x @inf", &x.ln() / &x, inf.clone()),
        ("x^2 e^(-x) @inf", x.powi(2) * (-&x).exp(), inf.clone()),
        (
            "(x^3 - 1)/(x - 1) @1",
            (x.powi(3) - 1) / (&x - 1),
            ctx.int(1),
        ),
        ("a x / x @0", &a * &x / &x, zero.clone()),
        ("(a^x - 1)/x @0", (a.pow(&x) - 1) / &x, zero.clone()),
        ("sin(ax)/x @0", (&a * &x).sin() / &x, zero.clone()),
        (
            "(sqrt(1+x)-1)/x @0",
            ((1 + &x).sqrt() - 1) / &x,
            zero.clone(),
        ),
        (
            "(cos x - 1)/(x sin x) @0",
            (&x.cos() - 1) / (&x * &x.sin()),
            zero.clone(),
        ),
        ("x^2 sin(1/x) @0", x.powi(2) * (1 / &x).sin(), zero.clone()),
        ("e^(-x^2) @inf", (-x.powi(2)).exp(), inf.clone()),
        (
            "(3x^2+1)/(x^2-x) @inf",
            (&x.powi(2) * 3 + 1) / (x.powi(2) - &x),
            inf.clone(),
        ),
        (
            "x - sqrt(x^2 - 1) @inf",
            &x - (x.powi(2) - 1).sqrt(),
            inf.clone(),
        ),
        ("ln(x+1) - ln(x) @inf", (&x + 1).ln() - x.ln(), inf.clone()),
        (
            "x(ln(x+1) - ln x) @inf",
            &x * ((&x + 1).ln() - x.ln()),
            inf.clone(),
        ),
        ("asin(x)/x @0", x.asin() / &x, zero.clone()),
        ("sinh(x)/x @0", x.sinh() / &x, zero.clone()),
        (
            "(cosh x - 1)/x^2 @0",
            (x.cosh() - 1) / x.powi(2),
            zero.clone(),
        ),
        ("Gamma(x) @inf", x.gamma(), inf.clone()),
        ("1/Gamma(x) @inf", 1 / x.gamma(), inf.clone()),
        ("x! / x^x @inf", x.factorial() / x.pow(&x), inf.clone()),
        ("e^x/x^x @inf", x.exp() / x.pow(&x), inf.clone()),
        (
            "(1 + 1/x)^(x^2) @inf",
            (1 + 1 / &x).pow(&x.powi(2)),
            inf.clone(),
        ),
        ("x^(1/x) @0", x.pow(&(1 / &x)), zero.clone()),
        ("sec x @0", x.sec(), zero.clone()),
        (
            "(1 - cos(2x))/x^2 @0",
            (1 - (&x * 2).cos()) / x.powi(2),
            zero.clone(),
        ),
        ("cos(x)/x @inf", x.cos() / &x, inf.clone()),
        ("x^3 @-inf", x.powi(3), ninf.clone()),
        ("e^x x^3 @-inf", x.exp() * x.powi(3), ninf.clone()),
        ("sin x @1", x.sin(), ctx.int(1)),
        ("e^x + pi @0", x.exp() + ctx.pi(), zero.clone()),
        ("sinh(x)/e^x @inf", x.sinh() / x.exp(), inf.clone()),
        ("x tanh x @inf", &x * x.tanh(), inf.clone()),
        ("atan(1/x) @0", (1 / &x).atan(), zero.clone()),
        ("sqrt(x) @0", x.sqrt(), zero.clone()),
        ("x/|x| @0", &x / &x.abs(), zero.clone()),
        ("a x @inf", &a * &x, inf.clone()),
        ("(x+a)/(x-a) @inf", (&x + &a) / (&x - &a), inf.clone()),
        ("Gamma(x) @3", x.gamma(), ctx.int(3)),
        ("asinh(x) - ln(x) @inf", x.asinh() - x.ln(), inf.clone()),
        ("e^x/(e^x+1) @inf", x.exp() / (x.exp() + 1), inf.clone()),
        ("e^x/(e^x+1) @-inf", x.exp() / (x.exp() + 1), ninf.clone()),
        ("(1+x)^(1/x) @0", (1 + &x).pow(&(1 / &x)), zero.clone()),
        ("ln(x)/(x-1) @1", x.ln() / (&x - 1), ctx.int(1)),
        ("(x-1)/ln(x) @1", (&x - 1) / x.ln(), ctx.int(1)),
        ("x^2 ln(x) @0", x.powi(2) * x.ln(), zero.clone()),
        (
            "ln(x)/x^(1/3) @inf",
            x.ln() / x.pow(&ctx.rational(1, 3)),
            inf.clone(),
        ),
        ("x^2 - x^3 @inf", x.powi(2) - x.powi(3), inf.clone()),
        ("x e^x @-inf", &x * x.exp(), ninf.clone()),
        ("(e^x - 1)/sin(x) @0", (x.exp() - 1) / x.sin(), zero.clone()),
        ("|sin x|/x @0", x.sin().abs() / &x, zero.clone()),
        (
            "cos(x)^(1/x^2) @0",
            x.cos().pow(&(1 / x.powi(2))),
            zero.clone(),
        ),
        (
            "(1 + sin x)^(1/x) @0",
            (1 + x.sin()).pow(&(1 / &x)),
            zero.clone(),
        ),
    ];
    let mut n_uneval = 0;
    for (label, e, pt) in cases {
        let r = e.limit(&x, &pt);
        let flag = if r.has_unevaluated() {
            n_uneval += 1;
            "UNEVAL"
        } else {
            "ok"
        };
        println!("{label:32} -> {flag:6} {r}");
    }
    println!("unevaluated: {n_uneval}");

    println!("--- one-sided ---");
    let pw = Ex::piecewise(&[
        (&x.powi(2), &x.lt(&ctx.int(1))),
        (&(&x * 2), &ctx.int(1).ge(&x)),
    ]);
    let one_sided: Vec<(&str, Ex, Ex)> = vec![
        ("1/x @0", 1 / &x, zero.clone()),
        ("|x|/x @0", &x.abs() / &x, zero.clone()),
        ("ln x @0", x.ln(), zero.clone()),
        ("e^(1/x) @0", (1 / &x).exp(), zero.clone()),
        ("floor(x) @2", x.floor(), ctx.int(2)),
        ("ceil(x) @2", x.ceiling(), ctx.int(2)),
        ("tan x @pi/2", x.tan(), ctx.pi() / 2),
        ("x^x @0", x.pow(&x), zero.clone()),
        ("sin(1/x) @0", (1 / &x).sin(), zero.clone()),
        ("x ln x @0", &x * &x.ln(), zero.clone()),
        ("H(x) @0", x.heaviside(), zero.clone()),
        ("sign(x) @0", x.sign(), zero.clone()),
        ("1/(1+e^(1/x)) @0", 1 / (1 + (1 / &x).exp()), zero.clone()),
        ("piecewise @1", pw, ctx.int(1)),
        ("x^(1/x) @0", x.pow(&(1 / &x)), zero.clone()),
        ("1/(x-1)^2 @1", 1 / (&x - 1).powi(2), ctx.int(1)),
        ("1/(x-1)^3 @1", 1 / (&x - 1).powi(3), ctx.int(1)),
        ("atan(1/x) @0", (1 / &x).atan(), zero.clone()),
        ("e^(-1/x)/x @0", (-1 / &x).exp() / &x, zero.clone()),
        ("sqrt(x) @0", x.sqrt(), zero.clone()),
        ("ln(x) x @0", x.ln() * &x, zero.clone()),
        ("Gamma(x) @0", x.gamma(), zero.clone()),
        ("x/|x| @0", &x / &x.abs(), zero.clone()),
        ("|x-1|/(x-1) @1", (&x - 1).abs() / (&x - 1), ctx.int(1)),
        ("sign(x^2 - 1) @1", (x.powi(2) - 1).sign(), ctx.int(1)),
        ("ln(x-a) @a", (&x - &a).ln(), a.clone()),
    ];
    for (label, e, pt) in one_sided {
        let r = e.limit_right(&x, &pt);
        let l = e.limit_left(&x, &pt);
        println!("{label:24} -> right: {r:<28} left: {l}");
    }
}
