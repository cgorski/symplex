//! Tests of session 15 (s15j4): float evidence in the domains.
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

/// `(X − 1/10)³` for `X ~ Uniform(0, 1)`: the inverse of the map on the
/// image `[−1/1000, 729/1000]` is the real cube root `1/10 + ∛y`.  Up to
/// 0.54 the principal `y^(1/3) + 1/10` was accepted because it inverted the
/// map at the two probes `x = 1/4, 3/4` (an `f64` comparison with a `10⁻⁹`
/// tolerance), so below `x = 1/10` the density at `y = −1/2000` was
/// `−26.4567 − 45.8243i` and the CDF `0.139685 + 0.068736i`.  Oracle
/// (exact): `F(y) = 1/10 + sign(y)·|y|^{1/3}`, `f(y) = |y|^{−2/3}/3`;
/// mpmath (30 and 60 digits): `F(−1/2000) = 0.020629947401590026262`,
/// `f(−1/2000) = 52.913368398939982492`.
#[test]
fn odd_power_of_a_shifted_uniform_has_a_real_density() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let g = (&x - ctx.rational(1, 10)).powi(3);
    let d = Distribution::uniform(ctx.int(0), ctx.int(1)).transformed(&x, &g)?;
    let y = ctx.rational(-1, 2000);
    // `eval_f64` refuses a non-real value.
    close(
        d.density(&y).eval_f64()?,
        52.913_368_398_939_98,
        1e-14,
        "pdf",
    );
    close(
        d.cdf(&y).eval_f64()?,
        0.020_629_947_401_590_026,
        1e-14,
        "cdf",
    );
    // Above x = 1/10 nothing changed: F(1/1000) = 1/10 + 1/10.
    assert_eq!(d.cdf(&ctx.rational(1, 1000)).eval(), ctx.rational(1, 5));
    Ok(())
}

/// The same map on `Exponential(1)` (a half-line support, probes at
/// `1/2` and `2` up to 0.54) and `2(X − 1/10)³ + 5` on `Uniform(0, 1)`:
/// the CDF below the inflection was complex (`0.132421 + 0.059728i` at
/// `y = −1/2000` for the exponential).  Oracle (exact, mpmath 30/60
/// digits): `1 − e^{−(1/10 + ∛y)}` at `y = −1/2000` is
/// `0.020418605853421675636`; `1/10 + ∛((y − 5)/2)` at `y = 4.9995` is
/// `0.037003947505256341762`.
#[test]
fn odd_power_maps_use_the_real_root_below_the_inflection() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let shifted = (&x - ctx.rational(1, 10)).powi(3);
    let e = Distribution::exponential(ctx.int(1)).transformed(&x, &shifted)?;
    close(
        e.cdf(&ctx.rational(-1, 2000)).eval_f64()?,
        0.020_418_605_853_421_676,
        1e-14,
        "Exp(1): cdf",
    );
    let affine = 2 * &shifted + 5;
    let u = Distribution::uniform(ctx.int(0), ctx.int(1)).transformed(&x, &affine)?;
    close(
        u.cdf(&ctx.rational(49_995, 10_000)).eval_f64()?,
        0.037_003_947_505_256_34,
        1e-14,
        "2(X − 1/10)³ + 5: cdf",
    );
    Ok(())
}

/// `sinh(X + 5)³` for a standard normal `X`: `solve`'s inverse
/// `asinh(∛y) − 5` takes the principal cube root, complex for `y < 0`, and
/// is not the inverse there.  Up to 0.54 it passed the four probes
/// `x = ±1/2, ±2` (where `y > 0`) and the density at `y = −1` was
/// `1.0267·10⁻⁵ + 1.0400·10⁻⁵i`, the CDF `−8.64·10⁻⁶ − 3.99·10⁻⁶i` (truly
/// `Φ(asinh(−1) − 5) = 2.0344·10⁻⁹`, mpmath).  The inverse is now checked
/// to be real and continuous on the whole image, so the principal one is
/// not used: 0.55 refused the map, and since 0.56 (s15k3) the real cube
/// root `asinh(∛ȳ) − 5` takes its place (mpmath, 30 and 60 digits:
/// `F(−1) = 2.0343781878508443596·10⁻⁹`).
#[test]
fn a_principal_inverse_that_is_complex_on_part_of_the_image_is_not_used() -> Result<(), SymplexError>
{
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let g = (&x + 5).sinh().powi(3);
    let d = Distribution::normal(ctx.int(0), ctx.int(1)).transformed(&x, &g)?;
    // `eval_f64` refuses a non-real value.
    close(
        d.cdf(&ctx.int(-1)).eval_f64()?,
        2.034_378_187_850_844_4e-9,
        1e-13,
        "cdf",
    );
    Ok(())
}

/// The order statistics of a finite parent are enumerated over its values
/// in increasing order, which up to 0.54 came from their `f64` values: two
/// values with one double (`1` and `1 + 10⁻²⁰`; `2⁵³` and `2⁵³ + 1`)
/// kept the order given, and the minimum of two draws from `{1 + 10⁻²⁰: ½,
/// 1: ½}` put mass `3/4` on `1 + 10⁻²⁰` and `1/4` on `1`.  Oracle (exact,
/// fractions): the minimum is the smaller value unless both draws are the
/// larger, `P = 1 − (½)² = 3/4`; for masses `(1/3, 2/3)` on `(2⁵³ + 1,
/// 2⁵³)` it is `1 − (1/3)² = 8/9` on `2⁵³` (was `4/9`).
#[test]
fn order_statistics_of_values_with_one_double_are_ordered_exactly() -> Result<(), SymplexError> {
    use symplex::stats::order::minimum_of;
    let ctx = Context::new();
    let near = &ctx.one() + ctx.rational(1, 10).powi(20);
    let half = ctx.rational(1, 2);
    let d = Distribution::finite(&ctx, vec![(near.clone(), half.clone()), (ctx.one(), half)]);
    let m = minimum_of(&d, 2)?;
    assert_eq!(m.density(&ctx.one()).eval(), ctx.rational(3, 4));
    assert_eq!(m.density(&near).eval(), ctx.rational(1, 4));
    let (big, bigger) = (
        ctx.int(9_007_199_254_740_992),
        ctx.int(9_007_199_254_740_993),
    );
    let d = Distribution::finite(
        &ctx,
        vec![
            (bigger.clone(), ctx.rational(1, 3)),
            (big.clone(), ctx.rational(2, 3)),
        ],
    );
    let m = minimum_of(&d, 2)?;
    assert_eq!(m.density(&big).eval(), ctx.rational(8, 9));
    assert_eq!(m.density(&bigger).eval(), ctx.rational(1, 9));
    Ok(())
}

/// The `f64` kernel of `quantile_f64` reads `n` of a binomial as a double:
/// up to 0.54 `Binomial(2⁵³ + 1, 1/2)` was taken for `Binomial(2⁵³, 1/2)`
/// (`2⁵³ + 1` rounds to `2⁵³`, inside the kernel's documented range `n ≤
/// 2⁵³`) and answered `2⁵²`.  An `n` the double does not represent is now
/// refused like any `n` beyond `2⁵³`; `2⁵³` itself still answers (its median
/// is `2⁵²`, exactly: the distribution is symmetric about it).
#[test]
fn binomial_quantile_refuses_an_n_the_double_does_not_hold() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    let big = Distribution::binomial(ctx.int(9_007_199_254_740_993), half.clone());
    assert!(matches!(
        big.quantile_f64(0.5),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let edge = Distribution::binomial(ctx.int(9_007_199_254_740_992), half);
    assert_eq!(edge.quantile_f64(0.5)?, 4_503_599_627_370_496.0);
    Ok(())
}
