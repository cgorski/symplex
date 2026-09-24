//! 0.25 — `(a^b)^c = a^(bc)` for every base when `c` is an integer, and
//! the display of powers that print as quotients.

use symplex::prelude::*;

#[test]
fn an_integer_power_of_a_rational_power_flattens_for_any_base() {
    // z^c is single-valued for integer c, so (a^b)^c = a^(bc) on the
    // principal branch for every complex a (SymPy's `Pow._eval_power` rule).
    // Before 0.25 only an integer b or a positive rational a flattened.
    let ctx = Context::new();
    let y = ctx.symbol("y");
    assert_eq!(y.sqrt().powi(2), y);
    assert_eq!(y.pow(&ctx.rational(1, 3)).powi(6), y.powi(2));
    assert_eq!(y.pow(&ctx.rational(2, 3)).powi(-3), y.powi(-2));
    let c = ctx.parse("-sin(-1)/2").unwrap();
    assert_eq!(c.sqrt().powi(2), c);
    // Numerically at a negative base, where a branch mistake would show:
    // ((-5)^(1/2))^3 = (-5)^(3/2) = -11.1803398874989484820458683437·i
    // (mpmath 1.3, dps 30: `power(sqrt(mpc(-5)), 3)` and
    // `power(mpc(-5), mpf(3)/2)` agree).
    let v = ctx.int(-5).sqrt().powi(3).eval_complex64().unwrap();
    assert!(
        v.re.abs() < 1e-12 && (v.im + 11.180_339_887_498_949).abs() < 1e-12,
        "{v}"
    );
}

#[test]
fn powers_that_print_as_quotients_are_parenthesised_and_round_trip() {
    let ctx = Context::new();
    for (src, shown) in [
        ("x^(1/y)", "x^(1/y)"),
        ("2^(1/y)", "2^(1/y)"),
        ("(1/y)^z", "(1/y)^z"),
        ("y^(-1/2)", "1/sqrt(y)"),
        ("a*(x+1)^(-1/2)", "a/sqrt(x + 1)"),
        ("(x+1)^(-1/2)*(y+1)^(-1/2)", "1/(sqrt(x + 1)*sqrt(y + 1))"),
        ("x - 2*(y+1)^(-1/2)", "x - 2/sqrt(y + 1)"),
        ("x^(y^(-1/2))", "x^(1/sqrt(y))"),
    ] {
        let e = ctx.parse(src).unwrap();
        assert_eq!(e.to_string(), shown, "{src}");
        // `x^(1/y)` printed as `x^1/y` before 0.25, which re-parses as x/y.
        assert_eq!(
            ctx.parse(shown).unwrap(),
            e,
            "{shown} does not re-parse to {src}"
        );
    }
}
