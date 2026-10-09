//! Exact evaluation (`eval`) of constants.

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `|e^z| = e^(Re z)` on all of ℂ.  Before: `abs(exp(I))` stayed
/// unevaluated, and the Frobenius norm of `[[exp(I)]]` printed
/// `sqrt(sin(1)^2 + cos(1)^2)`.  SymPy 1.14: `Abs(exp(I))` → `1`,
/// `Abs(exp(2 + 3*I))` → `exp(2)`, `Abs(exp(I*exp(I)))` → `exp(-sin(1))`,
/// `Matrix([[exp(I)]]).norm()` → `1`.
#[test]
fn modulus_of_an_exponential_is_the_exponential_of_the_real_part() {
    let ctx = Context::new();
    for (s, want) in [
        ("abs(exp(I))", "1"),
        ("abs(exp(2 + 3*I))", "exp(2)"),
        ("abs(exp(sqrt(2)*I))", "1"),
        ("abs(exp(I*exp(I)))", "exp(-sin(1))"),
    ] {
        assert_eq!(parse(&ctx, s).eval(), parse(&ctx, want).eval(), "{s}");
    }
    let m = Matrix::new(vec![vec![parse(&ctx, "exp(I)")]]).unwrap();
    assert_eq!(m.norm().eval(), ctx.int(1));
}

/// Binomials and factorial powers at negative or rational arguments, and
/// double factorials of negative odd integers, fold to their values.
/// Before: `binomial(-1, 3)`, `rising_factorial(3, -1)` and
/// `factorial2(-17)` stayed unevaluated, and `eval_decimal` of the first
/// two failed (`PrecisionExhausted`).  SymPy 1.14: `binomial(-1, 3)` → `-1`,
/// `binomial(Rational(7, 2), 2)` → `35/8`, `binomial(5, -1)` → `0`,
/// `rf(3, -1)` → `1/2`, `rf(3, -3)` → `zoo`, `rf(Rational(1, 2), -3)` →
/// `-8/15`, `ff(3, -1)` → `1/4`, `ff(-1, -1)` → `zoo`, `factorial2(-17)` →
/// `1/2027025`, `factorial2(-3)` → `-1`.
#[test]
fn binomials_and_factorial_powers_at_negative_arguments() {
    let ctx = Context::new();
    for (s, want) in [
        ("binomial(-1, 3)", "-1"),
        ("binomial(7/2, 2)", "35/8"),
        ("binomial(-1/2, 3)", "-5/16"),
        ("binomial(5, -1)", "0"),
        ("rising_factorial(3, -1)", "1/2"),
        ("rising_factorial(3, -3)", "zoo"),
        ("rising_factorial(1/2, -3)", "-8/15"),
        ("falling_factorial(3, -1)", "1/4"),
        ("falling_factorial(-1, -1)", "zoo"),
        ("factorial2(-17)", "1/2027025"),
        ("factorial2(-3)", "-1"),
    ] {
        assert_eq!(parse(&ctx, s).eval(), parse(&ctx, want), "{s}");
    }
    assert_eq!(
        parse(&ctx, "binomial(-1, 3)").eval_decimal(16).unwrap(),
        "-1"
    );
}

/// `(±1)^n` folds whatever the size of `n`.  Before: an exponent over the
/// power guard (1000) left `(-1)^430587161543285117552609` a power, and
/// `(-oo)*(-1)^430587161543285117552609` a product.  SymPy 1.14: `-1`, `oo`.
#[test]
fn powers_of_minus_one_fold_for_huge_exponents() {
    let ctx = Context::new();
    assert_eq!(parse(&ctx, "(-1)^430587161543285117552609"), ctx.int(-1));
    assert_eq!(parse(&ctx, "(-1)^(10^30)"), ctx.int(1));
    assert_eq!(
        parse(&ctx, "(-oo)*(-1)^430587161543285117552609"),
        parse(&ctx, "oo")
    );
}
