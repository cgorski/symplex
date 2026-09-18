//! symplex 0.3 — the univariate polynomial queries on `Ex` (`degree`,
//! `coeffs`, `coeff`, `leading_coeff`, `is_polynomial`) accept symbolic,
//! variable-free coefficients.  Rational-coefficient behaviour is pinned
//! byte-for-byte so the existing contract is unchanged.

use symplex::prelude::*;

fn s(e: &Ex) -> String {
    format!("{e}")
}

fn strs(v: &[Ex]) -> Vec<String> {
    v.iter().map(s).collect()
}

/// The motivating example: `j·r² + (j + 1)·r·f + 3` expanded.
fn motivating() -> (Context, Ex, Ex, Ex, Ex) {
    let ctx = Context::new();
    let (j, r, f) = (ctx.symbol("j"), ctx.symbol("r"), ctx.symbol("f"));
    let e = (&j * r.powi(2) + (&j + 1) * &r * &f + 3).expand();
    (ctx, j, r, f, e)
}

// ═══════════════════════════════════════════════════════════════════════════
// degree
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn degree_with_parameter_coefficients() {
    let (_ctx, _j, r, _f, e) = motivating();
    assert_eq!(e.degree(&r), Some(2));
}

#[test]
fn degree_in_the_parameter_itself() {
    let (_ctx, j, _r, f, e) = motivating();
    // f*j*r + j*r^2 + f*r + 3 is degree 1 in j and degree 1 in f.
    assert_eq!(e.degree(&j), Some(1));
    assert_eq!(e.degree(&f), Some(1));
}

#[test]
fn degree_unexpanded_symbolic_input_is_expanded_first() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let e = (&x + &a).powi(3) * (&x - 1);
    assert_eq!(e.degree(&x), Some(4));
}

#[test]
fn degree_symbolic_leading_terms_that_cancel_are_dropped() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    // a·x² − a·x² + x  →  degree 1
    let e = &a * x.powi(2) - &a * x.powi(2) + &x;
    assert_eq!(e.degree(&x), Some(1));
}

#[test]
fn degree_none_for_var_inside_function_with_parameters() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    assert_eq!((&a * x.sin()).degree(&x), None);
    assert_eq!((&a * x.exp() + &x).degree(&x), None);
}

#[test]
fn degree_none_for_negative_and_symbolic_powers() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    assert_eq!((&a / &x + 1).degree(&x), None);
    assert_eq!((x.pow(&a) + 1).degree(&x), None);
    assert_eq!((&a * x.sqrt()).degree(&x), None);
}

#[test]
fn degree_none_when_var_is_in_an_exponent() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    assert_eq!((a.pow(&x) + &x).degree(&x), None);
    assert_eq!(ctx.int(2).pow(&x).degree(&x), None);
}

#[test]
fn degree_of_transcendental_coefficient_is_fine() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // sin(y) does not involve x, so it is a legitimate coefficient.
    let e = y.sin() * x.powi(2) + ctx.pi() * &x + y.exp();
    assert_eq!(e.degree(&x), Some(2));
}

#[test]
fn degree_var_free_symbolic_constant_is_zero() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    assert_eq!((&a + 1).degree(&x), Some(0));
    assert_eq!(a.sin().degree(&x), Some(0));
}

// ═══════════════════════════════════════════════════════════════════════════
// coeffs / coeff
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn coeffs_with_parameter_coefficients() {
    let (_ctx, _j, r, _f, e) = motivating();
    let cs = e.coeffs(&r).unwrap();
    assert_eq!(strs(&cs), ["3", "f*j + f", "j"]);
}

#[test]
fn coeffs_are_ascending_and_zero_filled() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let e = &a * x.powi(3) + 1;
    let cs = e.coeffs(&x).unwrap();
    assert_eq!(strs(&cs), ["1", "0", "0", "a"]);
}

#[test]
fn coeffs_reconstruct_the_polynomial() {
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    let e = ((&a * &x + &b) * (&x - &a)).expand();
    let cs = e.coeffs(&x).unwrap();
    let mut rebuilt = ctx.zero();
    for (k, c) in cs.iter().enumerate() {
        rebuilt += c * x.powi(k as i64);
    }
    assert!((rebuilt - &e).expand().is_zero_structural());
}

#[test]
fn coeff_individual_powers_with_parameters() {
    let (ctx, j, r, f, e) = motivating();
    assert_eq!(e.coeff(&r, 2).unwrap(), j);
    assert_eq!(e.coeff(&r, 1).unwrap(), (&f * &j + &f).eval());
    assert_eq!(e.coeff(&r, 0).unwrap(), ctx.int(3));
    // Above the degree the coefficient is zero.
    assert_eq!(e.coeff(&r, 5).unwrap(), ctx.zero());
}

#[test]
fn coeffs_none_for_non_polynomial_with_parameters() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    assert!((&a * x.sin()).coeffs(&x).is_none());
    assert!((&a / &x).coeffs(&x).is_none());
    assert!((&a / &x).coeff(&x, 0).is_none());
}

#[test]
fn coeffs_of_symbolic_constant_is_single_entry() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let cs = (&a + 1).coeffs(&x).unwrap();
    assert_eq!(strs(&cs), ["a + 1"]);
}

// ═══════════════════════════════════════════════════════════════════════════
// leading_coeff
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn leading_coeff_with_parameters() {
    let (_ctx, j, r, _f, e) = motivating();
    assert_eq!(e.leading_coeff(&r).unwrap(), j);
}

#[test]
fn leading_coeff_is_a_sum_of_parameters() {
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    let e = (&a + &b) * x.powi(2) + &x;
    assert_eq!(s(&e.leading_coeff(&x).unwrap()), "a + b");
}

#[test]
fn leading_coeff_none_for_non_polynomial_with_parameters() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    assert!((&a * x.cos()).leading_coeff(&x).is_none());
    assert!((x.pow(&a)).leading_coeff(&x).is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// is_polynomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn is_polynomial_with_parameters() {
    let (_ctx, _j, r, _f, e) = motivating();
    assert!(e.is_polynomial(&r));
}

#[test]
fn is_polynomial_false_for_parametric_non_polynomials() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    assert!(!(&a * x.sin()).is_polynomial(&x));
    assert!(!(&a / &x).is_polynomial(&x));
    assert!(!(x.pow(&a)).is_polynomial(&x));
    assert!(!(a.pow(&x)).is_polynomial(&x));
}

#[test]
fn is_polynomial_rational_function_in_parameter_only() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    // 1/a is a fine coefficient; the expression is polynomial in x.
    assert!((x.powi(2) / &a + &x).is_polynomial(&x));
    // …but not in a.
    assert!(!(x.powi(2) / &a + &x).is_polynomial(&a));
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational-coefficient regression: unchanged behaviour
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rational_degree_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!((x.powi(3) + &x + 1).degree(&x), Some(3));
    assert_eq!((&x + 1).powi(4).degree(&x), Some(4));
    assert_eq!(ctx.int(7).degree(&x), Some(0));
    // The zero polynomial has no degree.
    assert_eq!(ctx.int(0).degree(&x), None);
    assert_eq!(x.sin().degree(&x), None);
    assert_eq!(x.powi(-1).degree(&x), None);
}

#[test]
fn rational_coeffs_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cs = (x.powi(2) + &x * 3 + 5).coeffs(&x).unwrap();
    assert_eq!(strs(&cs), ["5", "3", "1"]);
    let cs = (x.powi(3) * ctx.rational(1, 2) - 2).coeffs(&x).unwrap();
    assert_eq!(strs(&cs), ["-2", "0", "0", "1/2"]);
    // Zero polynomial → empty list.
    assert_eq!(ctx.int(0).coeffs(&x).unwrap().len(), 0);
}

#[test]
fn rational_coeff_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(2) * 3 + &x * 5 + 7;
    assert_eq!(s(&e.coeff(&x, 2).unwrap()), "3");
    assert_eq!(s(&e.coeff(&x, 1).unwrap()), "5");
    assert_eq!(s(&e.coeff(&x, 0).unwrap()), "7");
    assert_eq!(s(&e.coeff(&x, 9).unwrap()), "0");
}

#[test]
fn rational_leading_coeff_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(s(&(x.powi(3) * 5 - &x + 2).leading_coeff(&x).unwrap()), "5");
    assert_eq!(s(&ctx.int(0).leading_coeff(&x).unwrap()), "0");
    assert_eq!(s(&ctx.rational(-2, 3).leading_coeff(&x).unwrap()), "-2/3");
    assert!(x.sin().leading_coeff(&x).is_none());
}

#[test]
fn rational_is_polynomial_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!((x.powi(2) + 1).is_polynomial(&x));
    assert!(ctx.int(3).is_polynomial(&x));
    assert!(!x.sin().is_polynomial(&x));
    assert!(!x.powi(-2).is_polynomial(&x));
    assert!(!x.sqrt().is_polynomial(&x));
}

#[test]
fn rational_path_and_symbolic_path_agree_on_rational_input() {
    // Force the symbolic machinery through `as_poly` and compare with the
    // rational-only `coeffs` output.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x * 2 - 3).powi(3) * (&x + ctx.rational(1, 4));
    let via_coeffs = e.coeffs(&x).unwrap();
    let via_poly = e.as_poly(&[&x]).unwrap();
    let mut dense = via_poly.all_coeffs().unwrap();
    dense.reverse();
    assert_eq!(strs(&via_coeffs), strs(&dense));
}
