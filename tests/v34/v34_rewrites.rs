//! Rewrite routes on `0/0` at every point: a numerator that one of the
//! route's identities turns into 0, over a denominator that vanishes
//! identically by an identity of its functions (`tan x·cos x − sin x`,
//! `cosh²x − sinh²x − 1`, `sin²x + cos²x − 1`, `eˣ·e⁻ˣ − 1`).  `0` over
//! anything is `0` in the canonical arithmetic, so the denominator
//! disappeared and `together`, `expand_trig` and `trig_combine` returned a
//! finite value for an expression undefined everywhere, while `ratsimp`,
//! `expand`, `simplify`, `trigsimp` and `fu` already gave `nan` (found by
//! the rewrite-route hunter, 10,000 seeds of its base generator: 320
//! route disagreements, all of this kind, and by the undefined-expression
//! hunter: `together` 499 wrong of 10,000).  The oracle for each `nan` is
//! the definition: the denominator is 0 at every point (`mpmath` at x =
//! 16/13, 30 digits: `tan x·cos x − sin x` = `0.e-165` from SymPy's `N`),
//! the numerator is 0 once multiplied out, and `0/0` is `nan`.  SymPy 1.14
//! gives `0` from `together` (it finds the numerator 0 first), `x + 1` from
//! `expand_trig` of the first `expand_trig` case (it cancels `s/s`) and
//! `nan` for the second: symplex keeps `0/0` undefined, as its `ratsimp`
//! and `simplify` have since 0.32.

use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `together` of a quotient whose numerator cancels to 0 over a
/// denominator zero by an identity: `nan`, not `0`.  Before: `0`, `0`,
/// `(x² + 1)·(cos x + 2)` (the `0/0` term dropped), `y`, `zoo` for the
/// reciprocal (`(0/0)⁻¹`, a numerator zero by the identity over a
/// denominator that cancelled to 0), and `0` for `0/√s` (`s` zero by the
/// identity: `0·zoo`).
#[test]
fn together_keeps_zero_over_identity_zero_undefined() {
    let ctx = Context::new();
    for src in [
        "(x/(x+1) + 1/(x+1) - 1)/(tan(x)*cos(x) - sin(x))",
        "(-2*x/(x+2) + x*(-x/(x+2)+1))/(cosh(x)^2 - sinh(x)^2 - 1)",
        "(cos(x)+2)*(x^2+1) - (x/(x-1) + x*(-x/(x-1)+1))/(cos(2*x) - cos(x)^2 + sin(x)^2)",
        "y + (x/(x+1) + 1/(x+1) - 1)/(sin(x)^2 + cos(x)^2 - 1)",
        "((x/(x+1) + 1/(x+1) - 1)/(tan(x)*cos(x) - sin(x)))^(-1)",
        "(-x/(x+1) + x*(-x/(x+1)+1))/sqrt(cosh(x)^2 - sinh(x)^2 - 1)",
    ] {
        let e = p(&ctx, src);
        assert_eq!(e.together().to_string(), "nan", "together({src})");
        // The routes that already saw it agree.
        assert_eq!(e.ratsimp().to_string(), "nan", "ratsimp({src})");
    }
    // A denominator that does not vanish keeps the cancelled quotient 0.
    assert_eq!(
        p(&ctx, "y + (x/(x+1) + 1/(x+1) - 1)/(sin(x) + 2)")
            .together()
            .to_string(),
        "y"
    );
    assert_eq!(p(&ctx, "1/x + 1/y").together().to_string(), "(x + y)/(x*y)");
    // A polynomial zero denominator was already `nan` (0.31).
    assert_eq!(
        p(&ctx, "(x/(x+1) + 1/(x+1) - 1)/(x*(x+1) - x^2 - x) + 1")
            .together()
            .to_string(),
        "nan"
    );
}

/// A function whose argument is `zoo` at every point, beside a factor
/// that `eval` makes 0: `exp(1/s)·(exp(x/2)² − exp(x))` with `s = tan
/// x·cos x − sin x` is `exp(zoo)·0 = nan·0 = nan` at every point (SymPy
/// 1.14: `exp(zoo)` is `nan`, `sin(zoo)` is `nan`, `log(zoo)` is `zoo`;
/// `zoo·0` is `nan`), as symplex's `expand` gives.  Before, `simplify`,
/// `trigsimp` and `fu` gave `0`: every strategy starts with `eval`, which
/// folds `exp(x/2)²` and multiplies the `0` in (found by the
/// undefined-expression hunter, seed 14253).
#[test]
fn function_of_zoo_beside_a_vanishing_factor_is_undefined() {
    let ctx = Context::new();
    for src in [
        "exp(1/(tan(x)*cos(x)-sin(x)))*(exp(x/2)^2-exp(x))",
        "exp(1/(sin(x)^2+cos(x)^2-1))*(exp(x/2)^2-exp(x))",
        "sin(1/(tan(x)*cos(x)-sin(x)))*(exp(x/2)^2-exp(x))",
        "exp(1/(tan(x)*cos(x)-sin(x)))*(exp(2*x)-exp(x)^2)",
        "log(1/(tan(x)*cos(x)-sin(x)))*(exp(x/2)^2-exp(x))",
    ] {
        let e = p(&ctx, src);
        assert_eq!(e.simplify().to_string(), "nan", "simplify({src})");
        assert_eq!(e.expand().to_string(), "nan", "expand({src})");
    }
    let e = p(&ctx, "exp(1/(tan(x)*cos(x)-sin(x)))*(exp(x/2)^2-exp(x))");
    assert_eq!(e.simplify_trig().to_string(), "nan");
    assert_eq!(e.fu().to_string(), "nan");
    // A function of a finite argument beside the zero is 0, and nothing
    // changes without a vanishing factor.
    assert_eq!(
        p(&ctx, "exp(1/(sin(x)+2))*(exp(x/2)^2-exp(x))")
            .simplify()
            .to_string(),
        "0"
    );
    assert_eq!(
        p(&ctx, "exp(1/sin(x))*(x+1)").simplify().to_string(),
        "(x + 1)*exp(1/sin(x))"
    );
}

/// `expand_trig` and `trig_combine` whose identities make the numerator 0
/// over a denominator zero by an identity: `nan`, as `trigsimp` gives.
/// Before: `0` and `cos x + 2` (`expand_trig`), `0` and `0`
/// (`trig_combine`).
#[test]
fn expand_trig_and_trig_combine_keep_zero_over_identity_zero_undefined() {
    let ctx = Context::new();
    for src in [
        "(x+1)*(cos(2*x) - cos(x)^2 + sin(x)^2)/(sin(x)^2 + cos(x)^2 - 1)",
        "cos(x) + 2 + (sin(2*x) - 2*sin(x)*cos(x))/(exp(x)*exp(-x) - 1)",
    ] {
        let e = p(&ctx, src);
        assert_eq!(e.expand_trig().to_string(), "nan", "expand_trig({src})");
        assert_eq!(e.simplify_trig().to_string(), "nan", "trigsimp({src})");
    }
    for src in [
        "(x+1)*(sin(2*x) - 2*sin(x)*cos(x))/(cosh(x)^2 - sinh(x)^2 - 1)",
        "((cos(2*x) - cos(x)^2 + sin(x)^2)/(exp(x)*exp(-x) - 1))^2",
    ] {
        let e = p(&ctx, src);
        assert_eq!(e.trig_combine().to_string(), "nan", "trig_combine({src})");
        assert_eq!(e.simplify_trig().to_string(), "nan", "trigsimp({src})");
    }
    // Ordinary input is rewritten as before.
    assert_eq!(
        p(&ctx, "1/sin(2*x)").expand_trig().to_string(),
        "1/(2*sin(x)*cos(x))"
    );
    assert_eq!(
        p(&ctx, "(sin(2*x) - 2*sin(x)*cos(x))/(x+1)")
            .expand_trig()
            .to_string(),
        "0"
    );
    assert_eq!(
        p(&ctx, "2*sin(x)*cos(x)/(x+1)").trig_combine().to_string(),
        "sin(2*x)/(x + 1)"
    );
    assert_eq!(
        p(&ctx, "1/(sin(x)*cos(x))").trig_combine().to_string(),
        "1/(sin(x)*cos(x))"
    );
}
