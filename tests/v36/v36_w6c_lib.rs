//! Tests of the 0.50 refactor (w6c): library functions as Func nodes and the hand-written function lists.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

/// `Context::apply` with the name of a built-in function builds that
/// function, so it prints as text that parses back to it.
///
/// Before (0.49): `apply("sin", [x])` was an undefined function that printed
/// as `sin(x)`: `apply("sin", [x]) - x.sin()` displayed `-sin(x) + sin(x)`,
/// simplified to itself and re-parsed as `0`, and `eval_f64` of
/// `apply("sin", [1/2])` failed ("cannot evaluate function 'sin'");
/// `apply("C", [n, k])` was not `n.binomial(k)` although `C(n, k)` parses
/// as it.  Oracle: `sin(x) − sin(x) = 0`; mpmath `sin(0.5)` =
/// 0.479425538604203; `C(n, k)` is the display of the binomial coefficient.
#[test]
fn apply_with_a_builtin_name_builds_the_builtin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (n, k) = (ctx.symbol("n"), ctx.symbol("k"));
    let s = ctx.apply("sin", &[&x]).unwrap();
    assert_eq!(s, x.sin());
    assert_eq!((&s - &x.sin()).to_string(), "0");
    let half = ctx.apply("sin", &[ctx.rational(1, 2)]).unwrap();
    let v = half.eval_f64().unwrap();
    assert!((v - 0.479_425_538_604_203).abs() < 1e-15, "{v}");
    assert_eq!(ctx.apply("C", &[&n, &k]).unwrap(), n.binomial(&k));
    // Names compare ignoring ASCII case, and the parser's further names
    // count, as in the parser.
    assert_eq!(ctx.apply("log", &[&x]).unwrap(), x.ln());
    assert_eq!(
        ctx.apply("BesselJ", &[ctx.int(0), x.clone()]).unwrap(),
        ctx.parse("besselj(0, x)").unwrap()
    );
    // Every result prints as text that parses back to it.
    for e in [
        s,
        ctx.apply("C", &[&n, &k]).unwrap(),
        ctx.apply("sqrt", &[&x]).unwrap(),
    ] {
        assert_eq!(ctx.parse(&e.to_string()).unwrap(), e, "{e}");
    }
}

/// `apply("bell", [3])` is the library function, not an undefined function
/// that took on its meaning through its name.
///
/// Before (0.49): an `Apply` whose head was the text `bell`, which every
/// consumer resolved by name — `eval` gave 5 although the documentation
/// promised that nothing is known about an `apply`'d function.  Oracle:
/// Bell numbers B₃ = 5 (OEIS A000110), `ctx.int(3).bell()`.
#[test]
fn apply_of_a_library_name_is_the_library_function() {
    let ctx = Context::new();
    let b = ctx.apply("bell", &[ctx.int(3)]).unwrap();
    assert_eq!(b, ctx.int(3).bell());
    assert_eq!(b.eval(), ctx.int(5));
    assert_eq!(b.to_json().unwrap(), ctx.int(3).bell().to_json().unwrap());
}

/// A built-in name with an argument count the function does not take is an
/// error.
///
/// Before (0.49): `apply("Shi", [x, 3])` returned `Shi(x, 3)`, which every
/// consumer left alone and whose evaluation failed only later ("Shi called
/// with 2 argument(s)"); `apply("sin", [x, x])` returned `sin(x, x)`, which
/// does not parse.  Oracle: `Shi` takes one argument (DLMF 6.2.15), `sin`
/// one.
#[test]
fn apply_with_a_wrong_argument_count_is_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let err = ctx.apply("Shi", &[x.clone(), ctx.int(3)]).unwrap_err();
    assert!(matches!(err, SymplexError::InvalidArgument { .. }), "{err}");
    assert!(err.to_string().contains("Shi"), "{err}");
    assert!(ctx.apply("sin", &[&x, &x]).is_err());
    assert!(ctx.apply("besselj", &[&x]).is_err());
    // The internal names stay refused.
    assert!(ctx.apply("__ei_asym", &[&x]).is_err());
}

/// Any other name is still an undefined function: nothing is known about
/// it, and its derivative is formal.  `H` is one too (the parser reads
/// `H(x)` as Heaviside only while no undefined function `H` exists).
/// Oracle: the documented behaviour of `Context::apply` (unchanged).
#[test]
fn other_names_are_still_undefined_functions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.apply("f", &[&x]).unwrap();
    assert_eq!(f.to_string(), "f(x)");
    assert_eq!(f.eval(), f);
    assert_eq!(f.diff(&x).to_string(), "Derivative(f(x), x)");
    assert_eq!(f.expr_type(), ExprType::Apply);
    let h = ctx.apply("H", &[&x]).unwrap();
    assert_ne!(h, x.heaviside());
    assert_eq!(ctx.parse("H(x)").unwrap(), h);
}

/// `from_json` refuses an `Apply` headed by a reserved internal name and a
/// library function with an argument count it does not take.
///
/// Before (0.49): `{"type":"Apply","name":"__ei_asym",…}` built the limit
/// engine's internal function, whose limit at `∞` was `0` (W6a
/// found-not-fixed 1), and `besselj` with one argument built a node that no
/// consumer evaluates ("besselj called with 1 argument(s)").  Oracle: the
/// names are reserved (`Context::apply` and the parser refuse them);
/// `besselj(ν, x)` takes two arguments (DLMF 10.2.2).
#[test]
fn from_json_refuses_internal_heads_and_wrong_argument_counts() {
    let ctx = Context::new();
    let x = r#"{"type":"Symbol","name":"x"}"#;
    let internal = format!(r#"{{"type":"Apply","name":"__ei_asym","args":[{x}]}}"#);
    assert!(ctx.from_json(&internal).is_err());
    assert!(ctx.from_json_any(&internal).is_err());
    let bessel1 = format!(r#"{{"type":"Apply","name":"besselj","args":[{x}]}}"#);
    assert!(ctx.from_json(&bessel1).is_err());
    // The JSON form of a library function is its name and round-trips.
    let e = ctx
        .parse("besselj(0, x) + lambertw(x, -1) + bell(x)")
        .unwrap();
    let json = e.to_json().unwrap();
    assert!(json.contains(r#""name":"besselj""#), "{json}");
    assert_eq!(ctx.from_json(&json).unwrap(), e);
    // A built-in name in the `Apply` form is the built-in, as for `apply`.
    let sin = format!(r#"{{"type":"Apply","name":"sin","args":[{x}]}}"#);
    assert_eq!(ctx.from_json(&sin).unwrap(), ctx.symbol("x").sin());
}

/// Library functions keep their display, LaTeX, JSON, sort order and type
/// (byte identity of every printer; the golden harness checks 200,000
/// cases).  Oracle: the 0.49 outputs, recorded here.
#[test]
fn library_functions_print_as_before() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.parse("besselj(0, x) + erfi(x)").unwrap();
    assert_eq!(e.to_string(), "besselj(0, x) + erfi(x)");
    assert_eq!(
        e.to_latex(),
        r"J_{0}\left(x\right) + \operatorname{erfi}\left(x\right)"
    );
    assert_eq!(
        e.diff(&x).to_string(),
        "2*exp(x^2)/sqrt(pi) - 1/2*besselj(1, x) + 1/2*besselj(-1, x)"
    );
    let w = ctx.parse("lambertw(x, -1)").unwrap();
    assert_eq!(w.to_string(), "W(x, -1)");
    assert_eq!(w.expr_type(), ExprType::Apply);
    assert_eq!(w.diff(&x).to_string(), "W(x, -1)/(x*(W(x, -1) + 1))");
    let b = ctx.parse("bell(n) + harmonic(x)").unwrap();
    assert_eq!(b.to_latex(), "B_{n} + H_{x}");
}
