//! Regression tests for numeric radical canonicalisation (0.2 numfix):
//!
//! * `sqrt(3912608155036063)` took 14 s (trial division up to `√n` in
//!   `canon_pow`); square factors are now found with a bounded
//!   `factorint`, and composites beyond ~25 digits are left alone.
//! * `√(4/9)` was not folded, and `√(1/2)`, `1/√2`, `2^(-1/2)`, `√2/2`
//!   were four different expressions.  Positive rational bases are now
//!   split and denominators rationalised at construction (SymPy's form).

use std::time::{Duration, Instant};

use symplex::prelude::*;

fn s(e: &Ex) -> String {
    format!("{e}")
}

fn timed<T>(limit_ms: u64, what: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    let dt = t.elapsed();
    assert!(
        dt < Duration::from_millis(limit_ms),
        "{what} took {dt:?} (limit {limit_ms} ms)"
    );
    r
}

#[test]
fn sqrt_of_large_integer_with_square_factor_is_fast() {
    let ctx = Context::new();
    // 3912608155036063 = 7 · 23641997²
    let n = ctx.from_i128(3912608155036063);
    let r = timed(50, "sqrt(3912608155036063)", || n.sqrt().eval());
    assert_eq!(s(&r), "23641997*sqrt(7)");
}

#[test]
fn sqrt_of_thirty_digit_integer_is_fast() {
    let ctx = Context::new();
    // 343881576126216474609375000000 = 2⁶·3²·5¹⁶·7·23641997²
    let n = ctx.parse("343881576126216474609375000000").unwrap();
    let r = timed(50, "sqrt(30-digit)", || n.sqrt().eval());
    assert_eq!(s(&r), "221643721875000*sqrt(7)");
    let r = timed(50, "simplify sqrt(30-digit)", || n.sqrt().simplify());
    assert_eq!(s(&r), "221643721875000*sqrt(7)");
}

#[test]
fn sqrt_of_huge_composite_without_small_factors_is_left_alone_quickly() {
    let ctx = Context::new();
    // 4 · (10²⁰ + 39)(10²⁰ + 5559): only the 2² comes out; the 41-digit
    // semiprime is not factored at construction time.
    let n = ctx
        .parse("40000000000000002239200000000000000867204")
        .unwrap();
    let r = timed(100, "sqrt(41-digit semiprime)", || n.sqrt());
    assert_eq!(s(&r), "2*sqrt(10000000000000000559800000000000000216801)");
    // … and eval must not fall back to an unbounded search either.
    let e = timed(100, "eval", || r.eval());
    assert_eq!(e, r);
}

#[test]
fn sqrt_of_square_of_huge_prime_folds() {
    let ctx = Context::new();
    // 3 · (10¹⁹ + 51)²
    let n = ctx
        .parse("300000000000000003060000000000000007803")
        .unwrap();
    let r = timed(100, "sqrt(3p²)", || n.sqrt());
    assert_eq!(s(&r), "10000000000000000051*sqrt(3)");
}

#[test]
fn sqrt_12_is_2_sqrt_3() {
    let ctx = Context::new();
    let r = ctx.int(12).sqrt();
    assert_eq!(s(&r), "2*sqrt(3)");
    assert_eq!(r, ctx.int(2) * ctx.int(3).sqrt());
}

#[test]
fn sqrt_of_perfect_square_rational_folds() {
    let ctx = Context::new();
    assert_eq!(ctx.rational(4, 9).sqrt(), ctx.rational(2, 3));
    assert_eq!(ctx.rational(4, 9).sqrt().eval(), ctx.rational(2, 3));
    assert_eq!(
        ctx.rational(4, 9).pow(&ctx.rational(-1, 2)),
        ctx.rational(3, 2)
    );
    assert_eq!(
        ctx.rational(8, 27).pow(&ctx.rational(1, 3)),
        ctx.rational(2, 3)
    );
}

#[test]
fn sqrt_one_half_normal_form_unifies_all_constructions() {
    let ctx = Context::new();
    let want = ctx.rational(1, 2) * ctx.int(2).sqrt();
    assert_eq!(s(&want), "1/2*sqrt(2)");
    assert_eq!(ctx.rational(1, 2).sqrt(), want, "sqrt(1/2)");
    assert_eq!(ctx.int(2).sqrt() / ctx.int(2), want, "sqrt(2)/2");
    assert_eq!(ctx.int(1) / ctx.int(2).sqrt(), want, "1/sqrt(2)");
    assert_eq!(ctx.int(2).pow(&ctx.rational(-1, 2)), want, "2^(-1/2)");
    assert_eq!(ctx.parse("sqrt(1/2)").unwrap(), want, "parse sqrt(1/2)");
    assert_eq!(ctx.parse("2^(-1/2)").unwrap(), want, "parse 2^(-1/2)");
    assert_eq!(ctx.parse("1/sqrt(2)").unwrap(), want, "parse 1/sqrt(2)");
}

#[test]
fn rational_radicand_with_square_factor() {
    let ctx = Context::new();
    // √(8/3) = 2√6/3
    let r = ctx.rational(8, 3).sqrt();
    assert_eq!(s(&r), "2/3*sqrt(6)");
    // √(2/3) = √6/3
    assert_eq!(s(&ctx.rational(2, 3).sqrt()), "1/3*sqrt(6)");
    // (2/3)^(-1/2) = √6/2
    assert_eq!(
        s(&ctx.rational(2, 3).pow(&ctx.rational(-1, 2))),
        "1/2*sqrt(6)"
    );
}

#[test]
fn negative_fractional_exponent_beyond_minus_one() {
    let ctx = Context::new();
    // 2^(-3/2) = √2/4
    let r = ctx.int(2).pow(&ctx.rational(-3, 2));
    assert_eq!(s(&r), "1/4*sqrt(2)");
    // Value check.
    let v = r.eval_f64().unwrap();
    assert!((v - 2f64.powf(-1.5)).abs() < 1e-15, "{v}");
}

#[test]
fn cube_root_of_54_is_3_cbrt_2() {
    let ctx = Context::new();
    let r = ctx.int(54).pow(&ctx.rational(1, 3));
    assert_eq!(s(&r), "3*cbrt(2)");
    assert_eq!(r, ctx.int(3) * ctx.int(2).pow(&ctx.rational(1, 3)));
}

#[test]
fn negative_radicands() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(ctx.int(-4).sqrt(), ctx.int(2) * &i, "sqrt(-4)");
    assert_eq!(
        ctx.int(-12).sqrt(),
        ctx.int(2) * ctx.int(3).sqrt() * &i,
        "sqrt(-12)"
    );
    assert_eq!(
        ctx.rational(-4, 9).sqrt(),
        ctx.rational(2, 3) * &i,
        "sqrt(-4/9)"
    );
    assert_eq!(
        ctx.rational(-1, 2).sqrt(),
        ctx.rational(1, 2) * ctx.int(2).sqrt() * &i,
        "sqrt(-1/2)"
    );
}

#[test]
fn radical_normal_form_is_numerically_faithful() {
    let ctx = Context::new();
    for (p, q, a, b) in [
        (1i64, 2i64, 1i64, 2i64),
        (2, 3, -1, 2),
        (8, 3, 1, 2),
        (2, 3, 1, 3),
        (5, 7, -2, 3),
        (45, 8, 3, 2),
        (12, 1, 3, 2),
        (2, 1, -3, 2),
        (1000, 1, 2, 3),
    ] {
        let e = ctx.rational(p, q).pow(&ctx.rational(a, b));
        let got = e.eval_f64().unwrap();
        let want = (p as f64 / q as f64).powf(a as f64 / b as f64);
        assert!(
            ((got - want) / want).abs() < 1e-14,
            "({p}/{q})^({a}/{b}) = {e} = {got}, want {want}"
        );
    }
}

#[test]
fn products_of_radicals_still_combine() {
    let ctx = Context::new();
    assert_eq!(ctx.int(2).sqrt() * ctx.rational(1, 2).sqrt(), ctx.int(1));
    assert_eq!(ctx.int(2).sqrt() * ctx.int(3).sqrt(), ctx.int(6).sqrt());
    assert_eq!(
        ctx.int(6).sqrt() * ctx.int(3).sqrt(),
        ctx.int(3) * ctx.int(2).sqrt()
    );
    assert_eq!(ctx.int(2).sqrt() * ctx.int(2).sqrt(), ctx.int(2));
}
