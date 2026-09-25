//! Tests for assumption query methods (Wave Q) and complex functions (Wave O).

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Wave Q — Assumption query methods
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn is_even_for_known_even() {
    let ctx = Context::new();
    let x = ctx.symbol("x").assume(Assumption::Even).unwrap();
    assert_eq!(x.is_even(), Some(true));
    // even implies not odd
    assert_eq!(x.is_odd(), Some(false));
}

#[test]
fn is_odd_for_known_odd() {
    let ctx = Context::new();
    let x = ctx.symbol("x").assume(Assumption::Odd).unwrap();
    assert_eq!(x.is_odd(), Some(true));
    // odd implies not even
    assert_eq!(x.is_even(), Some(false));
}

#[test]
fn is_even_odd_for_integer_literals() {
    let ctx = Context::new();
    assert_eq!(ctx.int(4).is_even(), Some(true));
    assert_eq!(ctx.int(4).is_odd(), Some(false));
    assert_eq!(ctx.int(7).is_even(), Some(false));
    assert_eq!(ctx.int(7).is_odd(), Some(true));
    // zero is even
    assert_eq!(ctx.int(0).is_even(), Some(true));
    assert_eq!(ctx.int(0).is_odd(), Some(false));
}

#[test]
fn is_prime_for_literal() {
    let ctx = Context::new();
    assert_eq!(ctx.int(7).is_prime(), Some(true));
    assert_eq!(ctx.int(4).is_prime(), Some(false));
    assert_eq!(ctx.int(2).is_prime(), Some(true));
    // 1 is neither prime nor composite; the system returns None
    assert_eq!(ctx.int(1).is_prime(), None);
}

#[test]
fn is_composite_for_literal() {
    let ctx = Context::new();
    assert_eq!(ctx.int(4).is_composite(), Some(true));
    assert_eq!(ctx.int(9).is_composite(), Some(true));
    assert_eq!(ctx.int(7).is_composite(), Some(false));
}

#[test]
fn is_transcendental_for_pi() {
    let ctx = Context::new();
    assert_eq!(ctx.pi().is_transcendental(), Some(true));
    // transcendental implies not algebraic
    assert_eq!(ctx.pi().is_algebraic(), Some(false));
}

#[test]
fn is_irrational_for_pi() {
    let ctx = Context::new();
    assert_eq!(ctx.pi().is_irrational(), Some(true));
    // irrational implies not rational
    assert_eq!(ctx.pi().is_rational(), Some(false));
}

#[test]
fn is_algebraic_for_rational() {
    let ctx = Context::new();
    assert_eq!(ctx.rational(1, 3).is_algebraic(), Some(true));
    // rational numbers are not transcendental
    assert_eq!(ctx.rational(1, 3).is_transcendental(), Some(false));
}

#[test]
fn is_algebraic_for_integer() {
    let ctx = Context::new();
    assert_eq!(ctx.int(5).is_algebraic(), Some(true));
    assert_eq!(ctx.int(5).is_irrational(), Some(false));
}

#[test]
fn is_hermitian_for_real_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x").assume(Assumption::Real).unwrap();
    // real values are hermitian
    assert_eq!(x.is_hermitian(), Some(true));
}

#[test]
fn is_even_unknown_for_bare_symbol() {
    // Use a fresh context to avoid cross-test pollution from the default context.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.is_even(), None);
    assert_eq!(x.is_odd(), None);
    assert_eq!(x.is_prime(), None);
    assert_eq!(x.is_composite(), None);
}

#[test]
fn is_hermitian_for_assumed_hermitian() {
    let ctx = Context::new();
    let x = ctx.symbol("x").assume(Assumption::Hermitian).unwrap();
    assert_eq!(x.is_hermitian(), Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave O — Complex functions: conjugate() and arg()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn conjugate_of_real_is_itself() {
    let ctx = Context::new();
    let x = ctx.symbol("x").assume(Assumption::Real).unwrap();
    let conj = x.conjugate();
    // For real x, im(x) = 0 so conjugate = re(x) - 0*i = x
    let s = format!("{}", conj.simplify());
    assert_eq!(s, "x", "conjugate of real should be itself, got: {s}");
}

#[test]
fn conjugate_of_pure_imaginary() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let three = ctx.int(3);
    // z = 3i
    let z = &three * &i;
    let conj = z.conjugate();
    let s = format!("{}", conj.simplify());
    // conjugate(3i) = -3i
    assert!(
        s == "-3*I" || s == "-3I" || s == "-(3*I)" || s == "-3·I",
        "conjugate(3i) should be -3i, got: {s}"
    );
}

#[test]
fn conjugate_of_complex_literal() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let three = ctx.int(3);
    let four = ctx.int(4);
    // z = 3 + 4i
    let z = &three + &(&four * &i);
    let conj = z.conjugate();
    let s = format!("{}", conj.simplify());
    // conjugate(3 + 4i) = 3 - 4i
    assert!(
        s.contains("3") && (s.contains("-4") || s.contains("- 4")),
        "conjugate(3+4i) should be 3-4i, got: {s}"
    );
}

#[test]
fn arg_of_positive_real() {
    let ctx = Context::new();
    // arg(1) = atan2(0, 1) = 0
    let one = ctx.int(1);
    let a = one.arg().eval();
    let s = format!("{a}");
    assert!(
        s == "0" || s == "atan2(0, 1)",
        "arg(1) should be 0, got: {s}"
    );
}

#[test]
fn arg_of_one_plus_i() {
    // arg(1 + i) = atan2(1, 1) = pi/4
    let ctx = Context::new();
    let one = ctx.int(1);
    let i = ctx.i_unit();
    let z = &one + &i;
    let a = z.arg().eval();
    let s = format!("{a}");
    // Accept symbolic pi/4 in any display format, or atan2(1, 1)
    assert!(
        s.contains("pi/4") || s.contains("π/4") || s.contains("1/4*pi") || s.contains("atan"),
        "arg(1+i) should be pi/4, got: {s}"
    );
}

#[test]
fn arg_first_quadrant() {
    let ctx = Context::new();
    // arg(1 + i) = π/4
    let z = &ctx.int(1) + &ctx.i_unit();
    let a = z.arg().eval();
    // Should be atan2(1, 1) = π/4
    let v = a.eval_f64().expect("evalf should succeed for arg(1+i)");
    assert!((v - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
}

#[test]
fn arg_second_quadrant() {
    let ctx = Context::new();
    // arg(-1 + i) = 3π/4
    let z = &ctx.int(-1) + &ctx.i_unit();
    let a = z.arg().eval();
    let v = a.eval_f64().expect("evalf should succeed for arg(-1+i)");
    assert!(
        (v - 3.0 * std::f64::consts::FRAC_PI_4).abs() < 1e-10,
        "arg(-1+i) should be 3π/4, got {v}"
    );
}

#[test]
fn arg_negative_real() {
    let ctx = Context::new();
    // arg(-1) = π
    let z = ctx.int(-1);
    let a = z.arg().eval();
    let v = a.eval_f64().expect("evalf should succeed for arg(-1)");
    assert!(
        (v - std::f64::consts::PI).abs() < 1e-10,
        "arg(-1) should be π, got {v}"
    );
}

#[test]
fn atan2_basic() {
    let ctx = Context::new();
    // atan2(1, 1) = π/4
    let one = ctx.int(1);
    let result = one.atan2(&one).eval();
    let v = result
        .eval_f64()
        .expect("evalf should succeed for atan2(1,1)");
    assert!((v - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
}

#[test]
fn atan2_on_axes() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let neg_one = ctx.int(-1);

    // atan2(0, 1) = 0
    assert_eq!(format!("{}", zero.atan2(&one).eval()), "0");
    // atan2(1, 0) = π/2
    let r = one.atan2(&zero).eval();
    let v = r.eval_f64().expect("evalf should succeed for atan2(1,0)");
    assert!(
        (v - std::f64::consts::FRAC_PI_2).abs() < 1e-10,
        "atan2(1,0) should be π/2, got {v}"
    );
    // atan2(0, -1) = π
    let r = zero.atan2(&neg_one).eval();
    let v = r.eval_f64().expect("evalf should succeed for atan2(0,-1)");
    assert!(
        (v - std::f64::consts::PI).abs() < 1e-10,
        "atan2(0,-1) should be π, got {v}"
    );
}
