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
