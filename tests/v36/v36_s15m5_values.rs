//! Tests of session 15 (s15m5): no decimal round trips inside the library.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

/// `Context::decimal_str("1e100000000")` formed `10^100000000` (a
/// 330-million-bit integer) and did not return within 30 s up to 0.57; it
/// is refused now, as the expression parser refuses a literal exponent
/// beyond `±10000` (oracle: `Context::parse` of the same text errors).
/// Literals within the range are exact as before.
#[test]
fn decimal_str_refuses_exponents_beyond_the_parsers_range() {
    let ctx = Context::new();
    assert!(ctx.decimal_str("1e100000000").is_err());
    assert!(ctx.decimal_str("-2.5e-100000000").is_err());
    assert!(ctx.parse("1e100000000").is_err());
    assert_eq!(ctx.decimal_str("-12.5e-3").unwrap().to_string(), "-1/80");
    let big = ctx.decimal_str("1e10000").unwrap();
    assert_eq!(big, ctx.parse("1e10000").unwrap());
}

/// `x − exp(10⁹) > 0` did not return within 30 s up to 0.57: the 30-digit
/// approximation of the critical point `exp(10⁹) ≈ 8.0·10^434294481` was
/// turned into an exact rational, `10^434294452` times its digits.  The
/// point is refused now (the inequality stays unsolved) instead of formed.
/// Oracle: none needed for the refusal; an inequality with a moderate
/// point, `x − exp(10) > 0`, is still `(exp(10), oo)` (exact).
#[test]
fn a_critical_point_with_a_huge_decimal_exponent_is_refused_not_formed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x - ctx.int(10).pow(&ctx.int(9)).exp();
    assert!(SetEx::reduce_inequalities(&[e.gt(&ctx.int(0))], &x).is_err());
    let s = (&x - ctx.int(10).exp()).solve_gt(&x).to_string();
    assert_eq!(s, "(exp(10), oo)");
}

/// `x + i·e⁻⁶⁰ > 0` was `(-exp(-60)*I, oo)` up to 0.57: the sign chart
/// read the 30-digit printout of each value and took an imaginary part
/// below `10⁻²⁰` of the real part for 0, so the non-real `−i·e⁻⁶⁰` became
/// a real critical point and `p + i·e⁻⁶⁰` a positive sample.  The
/// imaginary part `evalf` keeps is certified nonzero (its ball excludes 0):
/// the expression is real at no real `x`, so the set is empty (oracle:
/// exact, `im(x + i·e⁻⁶⁰) = e⁻⁶⁰ > 0` for real `x`).
#[test]
fn a_certified_tiny_imaginary_part_makes_a_value_non_real() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x + ctx.i_unit() * ctx.int(-60).exp();
    assert_eq!(e.solve_gt(&x).to_string(), "EmptySet");
    let sol = SetEx::reduce_inequalities(&[e.gt(&ctx.int(0))], &x).unwrap();
    assert_eq!(sol.to_string(), "EmptySet");
    // A real expression with a tiny real part is unaffected.
    let tiny = &x - ctx.int(-60).exp();
    assert_eq!(tiny.solve_gt(&x).to_string(), "(exp(-60), oo)");
}
