//! Tests of session 15 (s15p3): the review queues of the 0.58 ratchets.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

use num_bigint::BigInt;
use symplex::combinatorics::{stirling1, stirling2};
use symplex::multipoly::MultiPoly;
use symplex::stats::Distribution;

/// `x^(2³¹)·y ± 1` as multivariate polynomials in `(x, y)`.
fn huge_pair() -> (MultiPoly, MultiPoly) {
    let x: MultiPoly = MultiPoly::var(2, 0);
    let y: MultiPoly = MultiPoly::var(2, 1);
    let one = MultiPoly::from_int(2, 1);
    let m = x.pow(1 << 31).mul(&y);
    (m.add(&one), m.sub(&one))
}

/// `MultiPoly::gcd(x^(2³¹)·y + 1, x^(2³¹)·y − 1)` panicked up to 0.58: the
/// modular routes refuse dense images of degree 2³¹ and the PRS fallback
/// multiplied by the leading coefficient `x^(2³¹)`, past the `u32`
/// exponents (the allowlisted assertion of `MultiPoly::mul`).  The PRS now
/// refuses: `try_gcd` is `None`, and `gcd` returns the gcd of the integer
/// contents, here 1 — which is the gcd (oracle, exact: the two differ by
/// 2, so a common divisor divides 2, and both have content 1).
#[test]
fn multipoly_gcd_with_exponents_near_2_pow_31_refuses_instead_of_panicking() {
    let (f, g) = huge_pair();
    assert_eq!(MultiPoly::try_gcd(&f, &g), None);
    assert_eq!(MultiPoly::gcd(&f, &g), MultiPoly::from_int(2, 1));
    // The product has the exponent 2³² of `x`: the lcm refuses too (was a
    // panic in `mul`).
    assert_eq!(MultiPoly::try_lcm(&f, &g), None);
    // Small inputs are unchanged: gcd((x + y)x, (x + y)y) = x + y.
    let x: MultiPoly = MultiPoly::var(2, 0);
    let y: MultiPoly = MultiPoly::var(2, 1);
    let s = x.add(&y);
    assert_eq!(MultiPoly::try_gcd(&s.mul(&x), &s.mul(&y)), Some(s));
}

/// `Ex::gcd_all` / `lcm_all` reached the same panic; they are `None` when
/// the gcd is refused (oracle: as above).
#[test]
fn gcd_all_of_huge_exponents_is_none_not_a_panic() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let m = x.pow(&ctx.int(1 << 31)) * &y;
    let f = &m + ctx.int(1);
    let g = &m - ctx.int(1);
    assert!(f.gcd_all(&g).is_none());
    assert!(f.lcm_all(&g).is_none());
}

/// `S(n, n − 2)` is small (`4·log₂ n` bits), so the size guard let
/// `stirling2(2⁴⁰, 2⁴⁰ − 2)` through, and the triangle sized its rows by
/// `k + 1 = 2⁴⁰ − 1` big integers (35 TB): the process aborted up to 0.58.
/// For `n = 2³¹` the explicit sum's powers `j^(2³¹)` (8 GB each) did the
/// same.  Beyond 2³² triangle cells both kinds refuse (`None`), at once;
/// values within the bounds are unchanged (oracle: SymPy
/// `stirling(10, 7) = 5880`, `stirling(10, 7, kind=1, signed=True) = -9450`).
#[test]
fn stirling_numbers_with_k_close_to_a_huge_n_are_refused_not_allocated() {
    let n = BigInt::from(1u64 << 40);
    let k: BigInt = &n - 2;
    assert_eq!(stirling2(n.clone(), k.clone()), None);
    assert_eq!(stirling1(n, k), None);
    let n31 = BigInt::from(1u64 << 31);
    let k31: BigInt = &n31 - 2;
    assert_eq!(stirling2(n31, k31), None);
    assert_eq!(stirling2(10, 7), Some(BigInt::from(5880)));
    assert_eq!(stirling1(10, 7), Some(BigInt::from(-9450)));
}

/// `stats::family::sign_of` read the sign of a constant from the first
/// character of its 5-digit printout and searched it for `I`, but `evalf`
/// prints the imaginary unit `i`: a non-real constant took the sign of its
/// real part, so `Uniform(0, 1).cdf(2 + i)` was `1`, `.sf(2 + i)` `0` and
/// `.cdf(−1 + i)` `0` up to 0.58.  The sign is read from the value
/// (`evalf::const_value`), and a certified nonzero imaginary part leaves it
/// unknown.  Oracle (exact): `2 + i` is not real, so `P(X ≤ 2 + i)` is not
/// decided by the support (SymPy raises "Invalid comparison of non-real").
/// Real points are unchanged: `cdf(3/2) = 1`, `cdf(1/2) = 1/2`.
#[test]
fn cdf_at_a_non_real_point_is_not_decided_from_the_real_part() {
    let ctx = Context::new();
    let u = Distribution::uniform(ctx.int(0), ctx.int(1));
    let two_plus_i = ctx.int(2) + ctx.i_unit();
    let minus_one_plus_i = ctx.int(-1) + ctx.i_unit();
    assert_ne!(u.cdf(&two_plus_i), ctx.int(1));
    assert_ne!(u.sf(&two_plus_i), ctx.int(0));
    assert_ne!(u.cdf(&minus_one_plus_i), ctx.int(0));
    assert_eq!(u.cdf(&ctx.rational(3, 2)), ctx.int(1));
    assert_eq!(u.cdf(&ctx.rational(1, 2)), ctx.rational(1, 2));
}

/// The certified quadrature of `integrate_numeric` parsed its 17-digit
/// decimal as an `f64`: a real value beyond `f64::MAX` became `inf` and was
/// reported "the integral is not a real number: 2.7263745721125666e347" up
/// to 0.58.  The `f64` is rounded from the value, and the refusal says the
/// value is beyond the range (oracle: `∫₋₁¹ e⁸⁰⁰·δ(x) dx = e⁸⁰⁰`, mpmath
/// `exp(800)` = 2.7263745721125665673e+347, real).  A value within range is
/// unchanged (`∫₋₁¹ 3δ(x) dx = 3`).
#[test]
fn certified_quadrature_beyond_the_f64_range_is_not_called_non_real() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (lo, hi) = (ctx.int(-1), ctx.int(1));
    let f = ctx.int(800).exp() * x.dirac_delta();
    let err = f.integrate_numeric(&x, &lo, &hi).unwrap_err().to_string();
    assert!(err.contains("beyond the f64 range"), "{err}");
    assert!(!err.contains("not a real number"), "{err}");
    let three = (ctx.int(3) * x.dirac_delta())
        .integrate_numeric(&x, &lo, &hi)
        .unwrap();
    assert_eq!(three, 3.0);
}
