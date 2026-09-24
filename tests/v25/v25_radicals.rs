//! 0.26 — radicals of integers keep a proper exponent, and inverse square
//! roots print as square roots in every printer.

use symplex::prelude::*;

#[test]
fn a_rational_power_above_one_splits_off_its_integer_part() {
    // SymPy 1.14 prints each the same way:
    //   2**Rational(3,2) → 2*sqrt(2);  Integer(15)**Rational(3,2) → 15*sqrt(15);
    //   Rational(29,15)**Rational(-3,2) → 15*sqrt(435)/841;
    //   2**Rational(5,3) → 2*2**(2/3);  Integer(3)**Rational(2,3) → 3**(2/3).
    let ctx = Context::new();
    for (src, shown) in [
        ("2^(3/2)", "2*sqrt(2)"),
        ("15^(3/2)", "15*sqrt(15)"),
        ("(29/15)^(-3/2)", "15/841*sqrt(435)"),
        ("2^(5/3)", "2*2^(2/3)"),
        ("3^(2/3)", "3^(2/3)"),
    ] {
        assert_eq!(ctx.parse(src).unwrap().to_string(), shown, "{src}");
    }
    // One number, one form: before 0.26 `15^(3/2)` and `15*sqrt(15)` were
    // different canonical forms and did not cancel.
    assert!(
        ctx.parse("15*sqrt(15) - 15^(3/2)")
            .unwrap()
            .is_zero_structural()
    );
}

#[test]
fn inverse_square_roots_render_as_square_roots_in_every_printer() {
    // With `(√b)^{-1}` flattened to `b^(-1/2)` (0.25), the LaTeX and pretty
    // printers showed `(x + 1)^{1/2}` in the denominator; they now match the
    // plain-text `a/sqrt(x + 1)`.
    let ctx = Context::new();
    let e = ctx.parse("a*(x+1)^(-1/2)").unwrap();
    assert_eq!(e.to_latex(), r"\frac{a}{\sqrt{x + 1}}");
    assert_eq!(
        ctx.parse("x^(-1/2)").unwrap().to_latex(),
        r"\frac{1}{\sqrt{x}}"
    );
    let pretty = e.pretty();
    assert!(pretty.contains("√(1 + x)"), "{pretty}");
    // A compound base in a product of denominators is parenthesised, and
    // the exponent shown is the positive one (it printed `x⁻²·y` for x²·y).
    let pretty = ctx.parse("a/((x+1)*y^2)").unwrap().pretty();
    assert!(
        pretty.contains("(1 + x)") && !pretty.contains('-'),
        "{pretty}"
    );
}
