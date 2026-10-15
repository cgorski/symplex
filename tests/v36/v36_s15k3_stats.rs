//! Tests of session 15 (s15k3): stats: transformed distributions.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;
use symplex::stats::Distribution;

fn close(actual: f64, expected: f64, rel: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{label}: {actual} vs {expected}"
    );
}

fn zero(e: &Ex, label: &str) {
    assert_eq!(e.simplify(), e.context().int(0), "{label}: {e}");
}

/// `ln X`, `ln(X + 1)`, `√X` and `X^(1/3)` are strictly monotone maps
/// whose inverse `solve` finds only for an image variable it knows real
/// (or positive): with the plain symbol `y` used up to 0.55 the roots held
/// under conditions (`eʸ` only when `im y ∈ (−π, π]`, `y²` only when `re y >
/// 0 ∨ …`), `solve` refused, and so did `transformed` (`NotImplemented`,
/// though the docs listed `ln x`).  Oracle (exact change of variables):
/// `ln X` for `X ~ Exp(1)` has the density `e^{y − eʸ}` and `F(0) = 1 −
/// e⁻¹`; `√X` has `2y·e^{−y²}`; `X^(1/3)` for `X ~ U(0, 1)` has `3y²` on
/// `[0, 1]`; `ln(X + 1)` has `F(1) = 1 − e^{1 − e}` (mpmath, 30 and 60
/// digits: `0.82062592126598281804`).
#[test]
fn logarithms_and_roots_of_a_positive_variable_are_transformed() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let exp1 = Distribution::exponential(ctx.int(1));
    let ln = exp1.transformed(&x, &x.ln())?;
    zero(&(ln.density(&y) - (&y - y.exp()).exp()), "ln X: density");
    assert_eq!(ln.cdf(&ctx.int(0)), 1 - ctx.int(-1).exp());
    let sqrt = exp1.transformed(&x, &x.sqrt())?;
    zero(
        &(sqrt.density(&y) - 2 * &y * (-y.powi(2)).exp()),
        "√X: density",
    );
    let unif = Distribution::uniform(ctx.int(0), ctx.int(1));
    let cbrt = unif.transformed(&x, &x.pow(&ctx.rational(1, 3)))?;
    zero(&(cbrt.density(&y) - 3 * y.powi(2)), "X^(1/3): density");
    assert_eq!(cbrt.cdf(&ctx.rational(1, 2)), ctx.rational(1, 8));
    let ln1 = exp1.transformed(&x, &(&x + 1).ln())?;
    close(
        ln1.cdf(&ctx.int(1)).eval_f64()?,
        0.820_625_921_265_982_8,
        1e-14,
        "ln(X + 1): cdf",
    );
    Ok(())
}

/// `−ln U` for a uniform `U` is `Exp(1)`, and `atan X` for a standard
/// normal `X` lives on `(−π/2, π/2)`: the roots `e^{−y}` and `tan y` hold
/// for every `y` of the image (for `tan y`: the condition `|y| < π/2` that
/// `solve` attaches is decided on the image).  Both were refused up to 0.55.
/// Oracle: `F(2) = 1 − e⁻²` exactly; `Φ(tan ½) = 0.70757098748316638740`
/// and the density `φ(tan ½)(1 + tan² ½) = 0.44619765275399121064` (mpmath,
/// 30 and 60 digits).
#[test]
fn conditions_of_the_roots_are_decided_on_the_image() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let unif = Distribution::uniform(ctx.int(0), ctx.int(1));
    let e = unif.transformed(&x, &(-x.ln()))?;
    assert_eq!(e.cdf(&ctx.int(2)), 1 - ctx.int(-2).exp());
    let n = Distribution::normal(ctx.int(0), ctx.int(1));
    let t = n.transformed(&x, &x.atan())?;
    let half = ctx.rational(1, 2);
    close(
        t.cdf(&half).eval_f64()?,
        0.707_570_987_483_166_4,
        1e-14,
        "atan X: cdf",
    );
    close(
        t.density(&half).eval_f64()?,
        0.446_197_652_753_991_2,
        1e-14,
        "atan X: pdf",
    );
    Ok(())
}

/// `X² + X` is strictly increasing on the support of `X ~ Exp(1)`, but
/// `solve` gives two real roots `(−1 ± √(1 + 4y))/2`, and up to 0.55 any
/// map with more than one real root was refused ("not a single branch").
/// Each root is now checked; the one that is real and continuous on the
/// image and inverts the map is taken.  Oracle: `F(3) = 1 − e^{−r}`, `r =
/// (√13 − 1)/2`, `= 0.72822360764960309690`; `f(3) = e^{−r}/(2r + 1) =
/// 0.075377209083075010531` (mpmath, 30 and 60 digits).
#[test]
fn the_inverse_is_chosen_among_several_real_roots() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d = Distribution::exponential(ctx.int(1)).transformed(&x, &(x.powi(2) + &x))?;
    let three = ctx.int(3);
    close(
        d.cdf(&three).eval_f64()?,
        0.728_223_607_649_603_1,
        1e-14,
        "cdf",
    );
    close(
        d.density(&three).eval_f64()?,
        0.075_377_209_083_075,
        1e-14,
        "pdf",
    );
    Ok(())
}

/// `exp((X − 1/10)³)` for `X ~ U(0, 1)` ran for more than 150 s up to 0.55
/// (then `sinh((X − 1/10)³)`, `exp(−(X − 1/10)³)` on `Exp(1)`): deciding
/// the monotonicity asked the inequality solver for `g′ ≥ 0`, whose sign
/// chart samples `x = ±1000`, and the decimal value of `exp(1000³) ≈
/// 10^(4.3·10⁸)` was turned into an exact rational.  The inverse is the
/// real cube root `1/10 + ∛(ln y)` (`solve`'s principal `∛(ln y)` is
/// complex for `y < 1`).  Oracle (mpmath, 30 and 60 digits): `F(2) = 1/10 +
/// (ln 2)^{1/3} = 0.98499704450051771875`, `f(2) = 1/(6 (ln 2)^{2/3}) =
/// 0.21279680788371436307`, and for `sinh((X − 1/10)³)`: `F(−1/2000) = 1/10
/// − asinh(1/2000)^{1/3} = 0.020629948503951759202`.
#[test]
fn an_exponential_of_an_odd_power_is_transformed_promptly() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let unif = Distribution::uniform(ctx.int(0), ctx.int(1));
    let cubed = (&x - ctx.rational(1, 10)).powi(3);
    let d = unif.transformed(&x, &cubed.exp())?;
    let two = ctx.int(2);
    close(
        d.cdf(&two).eval_f64()?,
        0.984_997_044_500_517_7,
        1e-14,
        "exp: cdf",
    );
    close(
        d.density(&two).eval_f64()?,
        0.212_796_807_883_714_36,
        1e-14,
        "exp: pdf",
    );
    let s = unif.transformed(&x, &cubed.sinh())?;
    close(
        s.cdf(&ctx.rational(-1, 2000)).eval_f64()?,
        0.020_629_948_503_951_76,
        1e-13,
        "sinh: cdf",
    );
    Ok(())
}

/// The sign chart of `exp(x³) ≥ 0` samples `x = ±1000`; the evaluated
/// `exp(±10⁹)` was parsed into an exact rational with a power of ten of
/// 434 294 482 digits, and `solve_ge` never returned (0.55).  The sign is
/// now read from the decimal's digits and exponent.  Oracle: `eᵗ > 0`.
#[test]
fn inequalities_with_huge_sample_values_are_decided() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.powi(3).exp().solve_ge(&x), ctx.reals());
    let d = 3 * (&x - ctx.rational(1, 10)).powi(2) * (&x - ctx.rational(1, 10)).powi(3).exp();
    assert_eq!(d.solve_ge(&x), ctx.reals());
}

/// `sinh(X + 5)³` for a standard normal `X`: `solve`'s `asinh(∛y) − 5` is
/// complex for `y < 0`; 0.55 refused the map (0.54 answered with complex
/// values).  The real cube root replaces the principal one — sound because
/// `asinh(ω∛y) − 5`, its value for `y < 0`, is one of `solve`'s roots.
/// Oracle (mpmath, 30 and 60 digits): `F(−1) = Φ(asinh(−1) − 5) =
/// 2.0343781878508443596·10⁻⁹`, `f(−1) = φ(asinh(−1) − 5)/(3 cosh(asinh
/// 1)) = 2.8975448928113194118·10⁻⁹`; `erf X` (inverse `erf⁻¹ y`, refused
/// up to 0.55): `F(½) = Φ(erf⁻¹ ½) = 0.68329625040803850903`.
#[test]
fn odd_roots_inside_functions_are_real_roots() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let n = Distribution::normal(ctx.int(0), ctx.int(1));
    let d = n.transformed(&x, &(&x + 5).sinh().powi(3))?;
    let m1 = ctx.int(-1);
    close(
        d.cdf(&m1).eval_f64()?,
        2.034_378_187_850_844_4e-9,
        1e-13,
        "sinh³: cdf",
    );
    close(
        d.density(&m1).eval_f64()?,
        2.897_544_892_811_319_4e-9,
        1e-13,
        "sinh³: pdf",
    );
    let e = n.transformed(&x, &x.erf())?;
    close(
        e.cdf(&ctx.rational(1, 2)).eval_f64()?,
        0.683_296_250_408_038_5,
        1e-14,
        "erf: cdf",
    );
    Ok(())
}
