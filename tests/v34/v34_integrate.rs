//! `∫ P·|g|`, `∫ P·sign(g)`, `∫ P·H(g)`: the antiderivative
//! `sign(g)·(G − G(r))` is continuous only when `r` runs over every real
//! root of `g`.  Before, `g` with a `sin`/`cos` of the variable took the
//! principal branches the solver returns for all of its roots, and the
//! answer jumped at the others.  Oracle: the derivative is not enough (it
//! is the integrand away from the jumps); `F(b) − F(a)` is compared with
//! mpmath `quad` over `[a, b]` split at the sign changes.

use symplex::prelude::*;

/// `sin(eˣ)` vanishes at `x = ln(kπ)` for every `k ≥ 1`; the solver
/// returns `ln π` alone, and `∫ sign(sin eˣ) dx` was
/// `(x − ln π)·sign(sin eˣ)`, which jumps by `2(ln(kπ) − ln π)` at
/// `ln(kπ)`: `F(2.93) − F(−2.31)` = 1.6694597716988 where mpmath
/// (`mp.dps = 30` and 50) `quad(lambda t: sign(sin(exp(t))), [-2.31,
/// log(pi), log(2*pi), log(3*pi), log(4*pi), log(5*pi), log(6*pi), 2.93])`
/// = 2.9266770905435486.  Also
/// `eˣ·|sin eˣ|` (`(−cos eˣ − 1)·sign(sin eˣ)`: 3.987653944 against
/// 11.98765394), `cos x·sign(sin eˣ)`, `sign(cos eˣ)` (`x − ln(π/2)`) and
/// `sign(sin e⁻ˣ)`.  SymPy 1.14 leaves them unevaluated
/// (`integrate(sign(sin(exp(x))), x)` is an `Integral` of a `Piecewise`);
/// so does symplex now.
#[test]
fn sign_of_a_periodic_argument_is_not_cut_at_one_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "sign(sin(exp(x)))",
        "exp(x)*abs(sin(exp(x)))",
        "cos(x)*sign(sin(exp(x)))",
        "1/(x^2+1)*sign(sin(exp(x)))",
        "sign(cos(exp(x)))",
        "sign(sin(exp(-x)))",
        "x*Heaviside(sin(exp(x)))",
    ] {
        let f = ctx.parse(src).unwrap();
        let big_f = f.integrate(&x);
        assert!(big_f.has_unevaluated(), "∫ {src} = {big_f}");
    }
}

/// `F(b) − F(a)` of the closed form, as an `f64`.
fn definite(ctx: &Context, src: &str, a: f64, b: f64) -> f64 {
    let x = ctx.symbol("x");
    let big_f = ctx.parse(src).unwrap().integrate(&x);
    assert!(!big_f.has_unevaluated(), "∫ {src} = {big_f}");
    let at = |t: f64| {
        big_f
            .subs(&x, &ctx.parse(&format!("{t}")).unwrap())
            .eval_f64()
            .unwrap()
    };
    at(b) - at(a)
}

/// The cases with all real roots known keep their closed forms, continuous
/// at the root: `eˣ − e⁻ˣ − 1` (one root `ln φ`, every root of the
/// quadratic in `eˣ` known), `x⁵ + x + 1 = (x² + x + 1)(x³ − x² + 1)` (one
/// real root `r`, the Sturm count of its real roots is 1) and `eˣ − 2`.
/// mpmath (`mp.dps = 30` and 50): `0.62 - 2*log((1+sqrt(5))/2)` =
/// −0.342423650119206895, the real root of `polyroots([1,-1,0,1])` =
/// −0.75487766624669276005, `0.62 - 2*r` = 2.1297553324933855201;
/// `quad(lambda t: abs(exp(t)-2)*t, [0, log(2), 2])` = 4.5773734045272718389.
#[test]
fn a_root_set_that_is_complete_keeps_its_closed_form() {
    let ctx = Context::new();
    let close = |got: f64, want: f64, what: &str| {
        assert!(
            (got - want).abs() < 1e-12 * want.abs().max(1.0),
            "{what}: {got} vs {want}"
        );
    };
    close(
        definite(&ctx, "sign(exp(x)-exp(-x)-1)", -2.31, 2.93),
        -0.342_423_650_119_206_9,
        "sign(eˣ − e⁻ˣ − 1)",
    );
    close(
        definite(&ctx, "sign(x^5+x+1)", -2.31, 2.93),
        2.129_755_332_493_385_6,
        "sign(x⁵ + x + 1)",
    );
    close(
        definite(&ctx, "x*abs(exp(x)-2)", 0.0, 2.0),
        4.577_373_404_527_272,
        "x·|eˣ − 2|",
    );
}
