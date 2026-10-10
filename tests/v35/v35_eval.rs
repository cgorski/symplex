//! Exact evaluation (`eval`) of constants.

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `Log z = ln|z| + i·Arg z` folds for constants whose argument is a
/// special rational multiple of π.  Before: `ln(I)`, `ln(1 + I)`, `ln(1/2 +
/// √3·i/2)` stayed unevaluated (and `solve(exp(I*x) - I)` returned
/// `−ln(I)·I`).  SymPy 1.14: `log(I)` → `I*pi/2`, `log(1 + I)` →
/// `log(sqrt(2)) + I*pi/4`, `log(3*I)` → `log(3) + I*pi/2`,
/// `log(1/2 + sqrt(3)*I/2)` → `I*pi/3`, `log(-sqrt(3)/2 - I/2)` →
/// `-5*I*pi/6`; mpmath `log(mpc(-2, -2))` = `1.03972077083992 −
/// 2.35619449019234j`.  `ln(2 + I)` (argument not a rational multiple of π)
/// and `ln(x + I)` stay.
#[test]
fn logarithms_of_special_complex_constants_fold() {
    let ctx = Context::new();
    for (s, want) in [
        ("ln(I)", "I*pi/2"),
        ("ln(-I)", "-I*pi/2"),
        ("ln(1 + I)", "ln(sqrt(2)) + I*pi/4"),
        ("ln(3*I)", "ln(3) + I*pi/2"),
        ("ln(-2 - 2*I)", "ln(2*sqrt(2)) - 3*I*pi/4"),
        ("ln(1/2 + sqrt(3)*I/2)", "I*pi/3"),
        ("ln(-sqrt(3)/2 - I/2)", "-5*I*pi/6"),
    ] {
        assert_eq!(parse(&ctx, s).eval(), parse(&ctx, want).eval(), "{s}");
    }
    for s in ["ln(2 + I)", "ln(x + I)"] {
        assert_eq!(parse(&ctx, s).eval(), parse(&ctx, s), "{s}");
    }
}
