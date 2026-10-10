//! Differentiation with respect to something other than a symbol.

use symplex::prelude::*;

/// `Ex::diff` with a non-symbol variable returned `0` for every input up
/// to 0.40: `(x²).diff(&(2x))` was `0`, and so was `(f(x)³).diff(&f(x))`
/// (the unit-aware calculus inherited it: `d(τ²)/d(Time::minutes(τ))` was
/// `0`).  SymPy 1.14 differentiates with respect to an undefined function
/// or a derivative of one, treating velocities as independent of their
/// coordinates (`diff(f(x)*Derivative(f(x), x) + Derivative(f(x), x)**2 +
/// x*Derivative(f(x), (x, 2)), f(x))` = `Derivative(f(x), x)`; with respect
/// to `Derivative(f(x), x)`: `f(x) + 2*Derivative(f(x), x)`), and refuses
/// anything else ("Can't calculate derivative wrt 2*x"): symplex leaves the
/// formal `Derivative`, which `try_diff` reports.
#[test]
fn derivatives_with_respect_to_function_values_and_refusals() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.apply("f", std::slice::from_ref(&x)).unwrap();
    let df = f.formal_diff(&x);
    let d2f = df.formal_diff(&x);

    let e = f.powi(3) + &x * &f + x.sin().powi(2);
    assert_eq!(e.diff(&f), 3 * f.powi(2) + &x);

    let lagrangian = &f * &df + df.powi(2) + &d2f * &x;
    assert_eq!(lagrangian.diff(&f), df);
    assert_eq!(lagrangian.diff(&df), &f + 2 * &df);

    for var in [2 * &x, &x + 1, x.sin()] {
        let d = x.powi(2).diff(&var);
        assert!(d.has_unevaluated(), "d/d({var}) x² = {d}");
        assert!(x.powi(2).try_diff(&var).is_err());
    }
    // A symbol still differentiates as before.
    assert_eq!(x.powi(3).diff(&x), 3 * x.powi(2));
}
