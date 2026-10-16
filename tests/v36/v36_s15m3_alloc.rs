//! Tests of session 15 (s15m3): bounded allocation lengths.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    match ctx.parse(s) {
        Ok(e) => e,
        Err(err) => panic!("{s}: {err}"),
    }
}

/// `Z⁻¹{z/(z − 1)^m} = C(n, m − 1)`, written as the falling factorial
/// `n(n − 1)⋯(n − m + 2)/(m − 1)!`.  Up to 0.57 `(m − 1)!` was an `i64`
/// product, which wraps from `m = 22`: `z/(z − 1)^22` gave
/// `−n(n − 1)⋯(n − 20)/4249290049419214848` (negative, and the wrong size).
/// Oracle: the transform pair `Σ_{n≥0} C(n, k) z^(−n) = z/(z − 1)^(k+1)`
/// (exact identity); `C(30, 21) = 14307150`, `C(21, 21) = 1`, and
/// `C(20, 21) = 0`.
#[test]
fn inverse_z_transform_of_a_pole_of_order_22() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let n = ctx.symbol("n");
    let x = parse(&ctx, "z/(z-1)^22")
        .inverse_z_transform(&z, &n)
        .unwrap();
    assert_eq!(x.subs(&n, &ctx.int(30)).eval(), ctx.int(14_307_150));
    assert_eq!(x.subs(&n, &ctx.int(21)).eval(), ctx.int(1));
    assert_eq!(x.subs(&n, &ctx.int(20)).eval(), ctx.int(0));
}

/// The same pair with `a = 2`: `Z⁻¹{z/(z − 2)^25} = C(n, 24)·2^(n − 24)`.
/// Up to 0.57 the coefficient was `−1/7835185981329244160` (`24!` wrapped
/// in `i64`).  Oracle: exact, `C(30, 24)·2⁶ = 593775·64 = 38001600`.
#[test]
fn inverse_z_transform_of_a_pole_of_order_25_at_two() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let n = ctx.symbol("n");
    let x = parse(&ctx, "z/(z-2)^25")
        .inverse_z_transform(&z, &n)
        .unwrap();
    assert_eq!(x.subs(&n, &ctx.int(30)).eval(), ctx.int(38_001_600));
}

/// A pole of huge order is refused.  Up to 0.57 `z/(z − 1)^(2³¹ − 1)`
/// collected the `m − 1` factors of `C(n, m − 1)` into a vector of `m`
/// slots (8 GB) and aborted the process, `z/(z − 1)^(2⁶³)` panicked
/// ("capacity overflow"), and `z/(z − 1)^65537`, `z/(z − 1)^100` gave `zoo`
/// (the `i64` factorial wrapped to 0).  Now orders above 10,000 (the degree
/// bound of the polynomial conversion) are an error.  Oracle: `C(n, m − 1)`
/// at `n = m − 1` is 1 and at `n = m` is `m` (exact).
#[test]
fn inverse_z_transform_refuses_a_huge_pole_order() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let n = ctx.symbol("n");
    for src in [
        "z/(z-1)^2147483647",
        "1/(z-1)^4294967297",
        "z/(z-1)^9223372036854775808",
        "z/(z-1)^65537",
        "z/(z-1)^10001",
    ] {
        let r = parse(&ctx, src).inverse_z_transform(&z, &n);
        assert!(r.is_err(), "{src}: not refused");
    }
    for m in [100i64, 1025] {
        let x = parse(&ctx, &format!("z/(z-1)^{m}"))
            .inverse_z_transform(&z, &n)
            .unwrap();
        assert_eq!(x.subs(&n, &ctx.int(m - 1)).eval(), ctx.int(1), "m = {m}");
        assert_eq!(x.subs(&n, &ctx.int(m)).eval(), ctx.int(m), "m = {m}");
    }
}

/// `fourier_series_on` reserved two coefficient vectors of `n_terms`
/// entries before computing any: up to 0.57 `n_terms = 2³¹ − 1` asked for
/// 34 GB and aborted the process.  Now more than `2²⁰` harmonics are
/// refused (`InvalidArgument`), and `fourier_series` keeps the harmonics a
/// formal `Sum` over a dummy index with formal coefficient integrals.
/// Oracle: the definition `aₙ = (1/π)∫_{−π}^{π} f cos(nx) dx`; for `f = x`,
/// `a₀ = 0` and the truncation to 3 harmonics is `2 sin x − sin 2x +
/// (2/3) sin 3x` (`bₙ = 2(−1)^(n+1)/n`).
#[test]
fn fourier_series_beyond_the_harmonic_cap() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let r = x.fourier_series_on(&x, &(-&pi), &pi, u32::MAX);
    assert!(
        matches!(r, Err(SymplexError::InvalidArgument { .. })),
        "{:?}",
        r.map(|s| s.a0)
    );
    let formal = x.fourier_series(&x, u32::MAX);
    assert!(formal.to_string().starts_with("Sum("), "{formal}");
    let three = x.fourier_series(&x, 3);
    let expected = parse(&ctx, "2*sin(x) - sin(2*x) + 2*sin(3*x)/3");
    assert!((three - expected).simplify().is_zero_structural());
}

/// `Poly::all_coeffs` laid out the dense list before looking at its
/// length: up to 0.57 `x^(2³¹ − 1) − 1` asked for 34 GB and aborted the
/// process.  Now `None` beyond degree `2²⁴`.  Oracle: the list of `x^65537 −
/// 1` has `65538` entries, the first `1` and the last `−1` (definition).
#[test]
fn all_coeffs_beyond_the_dense_cap() {
    use symplex::poly_ex::Poly;
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let huge = Poly::new(&parse(&ctx, "x^2147483647 - 1"), &[&x]).unwrap();
    assert!(huge.all_coeffs().is_none());
    let p = Poly::new(&parse(&ctx, "x^65537 - 1"), &[&x]).unwrap();
    let cs = p.all_coeffs().unwrap();
    assert_eq!(cs.len(), 65_538);
    assert_eq!(cs[0], ctx.int(1));
    assert_eq!(cs[65_537], ctx.int(-1));
}

/// `Σ_{k≥1} 1/k^p` for an even `p` is `ζ(p) = r·π^p` with `r` from the
/// Bernoulli number `B_p`.  Up to 0.57 the exponent had no bound: `p = 2⁵³`
/// asked the tangent-number table behind `B_p` for 144 PB and aborted the
/// process (`p = 65536` ran for minutes).  Beyond `p = 30000` the sum is
/// the node `ζ(p)`.  Oracle: the definition `ζ(s) = Σ_{k≥1} k^(−s)`, and
/// `ζ(4) = π⁴/90` (Euler) below the bound.
#[test]
fn p_series_with_a_huge_even_exponent() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let one = ctx.int(1);
    let oo = ctx.infinity();
    let s = parse(&ctx, "1/k^9007199254740992").summation(&k, &one, &oo);
    assert_eq!(s, parse(&ctx, "zeta(9007199254740992)"));
    let s4 = parse(&ctx, "1/k^4").summation(&k, &one, &oo);
    assert_eq!(s4, parse(&ctx, "pi^4/90"));
}

/// `primes_up_to`, `quadratic_residues` and `binomial_coefficients_list`
/// return a list as long as their input: up to 0.57 `primes_up_to(2³² + 1)`
/// (4 GB for the sieve), `quadratic_residues(2³¹ − 1)` (8 GB) and
/// `binomial_coefficients_list(2³¹ − 1)` aborted the process.  Now their
/// `try_` twins refuse beyond the documented bounds (`2³² − 1`, `2²⁶`,
/// `2¹⁶`) and the plain functions return an empty list there, as
/// `sqrt_mod_all` does.  Oracle: `π(10⁴) = 1229` (prime tables), the
/// residues modulo 7 are `{0, 1, 2, 4}` (SymPy `quadratic_residues(7)`),
/// `C(10, 5) = 252`.
#[test]
fn ntheory_lists_refuse_beyond_their_bounds() {
    use num_bigint::BigInt;
    use symplex::ntheory::{
        binomial_coefficients_list, primes_up_to, quadratic_residues,
        try_binomial_coefficients_list, try_primes_up_to, try_quadratic_residues,
    };
    assert_eq!(try_primes_up_to(10_000).unwrap().len(), 1229);
    assert!(try_primes_up_to((1 << 32) + 1).is_err());
    assert!(primes_up_to(1 << 53).is_empty());
    let b = |v: &[i64]| v.iter().map(|&x| BigInt::from(x)).collect::<Vec<_>>();
    assert_eq!(try_quadratic_residues(7).unwrap(), b(&[0, 1, 2, 4]));
    assert!(try_quadratic_residues((1u64 << 31) - 1).is_err());
    assert!(quadratic_residues(BigInt::from(1u64 << 62)).is_empty());
    assert_eq!(
        try_binomial_coefficients_list(10).unwrap()[5],
        BigInt::from(252)
    );
    assert!(try_binomial_coefficients_list((1 << 31) - 1).is_err());
    assert!(binomial_coefficients_list(u32::MAX).is_empty());
}

/// `∏_{k=−10}^{2⁶³−1} k`: up to 0.57 the enumeration test `b − a <
/// 1000` overflowed (a panic in a debug build; in a release build it
/// wrapped negative and the product asked for `2⁶³` slots — "capacity
/// overflow").  Now the difference is checked and the product goes to the
/// closed forms.  Oracle: the range holds `k = 0`, so a value, if any, is
/// `0`; `∏_{k=1}^{1000} k = 1000!` is still enumerated (the bound is
/// inclusive) and `∏_{k=1}^{1001} k = 1001!` comes from the closed form.
#[test]
fn product_over_a_range_wider_than_i64() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let p = k.product_over(&k, &ctx.int(-10), &parse(&ctx, "9223372036854775807"));
    assert!(p.is_zero_structural() || p.has_unevaluated(), "{p}");
    let p1000 = k.product_over(&k, &ctx.int(1), &ctx.int(1000));
    assert_eq!(p1000, parse(&ctx, "factorial(1000)").eval());
    let p1001 = k.product_over(&k, &ctx.int(1), &ctx.int(1001));
    assert_eq!(p1001, parse(&ctx, "factorial(1001)").eval());
}
