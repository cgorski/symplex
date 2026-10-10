//! Exact evaluation (`eval`) of constants, and expressions undefined at
//! every point.

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

/// `0·sin(1/d)` with `d` zero by an identity of its functions is `0·nan`:
/// `tan x·cos x − sin x = 0` wherever it is defined, so `sin(1/d) =
/// sin(zoo)` has no value at any point.  Before, `ratsimp` multiplied the
/// zero factor `x·(x + 2) − x² − 2x` out and gave `0` (`simplify` and
/// `expand` gave `nan`); SymPy 1.14 `ratsimp` gives `0` (symplex keeps `0/0`
/// and `0·nan` undefined, deliberately different).  A factor whose argument
/// has an ordinary pole still cancels to `0`.
#[test]
fn ratsimp_keeps_zero_times_a_function_of_a_pole_by_identity_undefined() {
    let ctx = Context::new();
    for s in [
        "sin(1/(tan(x)*cos(x) - sin(x)))*(x*(x + 2) - x^2 - 2*x)",
        "exp(1/(cosh(x)^2 - sinh(x)^2 - 1))*(x*(x + 3) - x^2 - 3*x)",
        "sin(1/(sin(x)^2 + cos(x)^2 - 1))*(x*(x + 2) - x^2 - 2*x)*y",
    ] {
        assert_eq!(parse(&ctx, s).ratsimp(), ctx.nan(), "{s}");
    }
    for s in [
        "sin(1/(x + 1))*(x*(x + 2) - x^2 - 2*x)",
        "sin(1/(tan(x)*cos(x) + sin(x)))*(x*(x + 2) - x^2 - 2*x)",
    ] {
        assert_eq!(parse(&ctx, s).ratsimp(), ctx.int(0), "{s}");
    }
}
