//! 0.22 — simplify regressions from the 0.21 differential audit.
//!
//! `pow_pow` (`(x^a)^b → x^(a·b)`) used to fire whenever *any* bound value
//! was an integer — including the inner exponent `a`, which is exactly the
//! unsafe case: `(x²)^(3/2) = |x|³ ≠ x³`.  The rule now mirrors SymPy's
//! `Pow._eval_power`: it fires iff the outer exponent is an integer, the
//! base is known non-negative, or the inner exponent is a real number in
//! `(−1, 1]`.
//!
//! Reference values cite sympy 1.14 (`symplex/.venv/bin/python`) by the
//! call that produced them (float literals are the shortest round-trip
//! `f64` of the quoted 17-digit sympy value, as clippy requires).  Nothing
//! below was computed by hand.

use symplex::prelude::*;

/// Value of `e` at `x = -2` (both `x` and `y` are substituted so the same
/// helper serves the `cbrt(y²)` case).
fn at_minus_two(e: &Ex) -> f64 {
    let ctx = e.context();
    let m2 = ctx.int(-2);
    e.subs(&ctx.symbol("x"), &m2)
        .subs(&ctx.symbol("y"), &m2)
        .eval_f64()
        .unwrap_or_else(|err| panic!("{e} at -2 did not evaluate to a real f64: {err}"))
}

/// `simplify(e)` must keep the value at `x = -2` (equal to `reference` from
/// sympy) and must not be the structurally collapsed form `collapsed`.
fn assert_not_collapsed(e: &Ex, collapsed: &Ex, reference: f64) {
    let s = e.simplify();
    assert_ne!(
        s.id(),
        collapsed.id(),
        "{e} must not collapse to {collapsed}; got {s}"
    );
    let orig = at_minus_two(e);
    let simp = at_minus_two(&s);
    assert!(
        (orig - reference).abs() <= 1e-12 * reference.abs().max(1.0),
        "{e} at -2: got {orig}, sympy says {reference}"
    );
    assert!(
        (simp - reference).abs() <= 1e-12 * reference.abs().max(1.0),
        "simplify({e}) = {s} at -2: got {simp}, sympy says {reference}"
    );
}

// ── Integer inner exponent, fractional outer: must be left alone ────────────

#[test]
fn pow_pow_x2_to_3_2_stays() {
    // sympy: simplify((x**2)**Rational(3,2)) == (x**2)**(3/2);
    //        ((x**2)**Rational(3,2)).subs(x, -2) == 8
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(2).pow(&ctx.rational(3, 2));
    assert_not_collapsed(&e, &x.powi(3), 8.0);
}

#[test]
fn pow_pow_sqrt_of_inverse_square_stays() {
    // sympy: simplify(sqrt(1/x**2)) == sqrt(x**(-2)); sqrt(1/x**2).subs(x, -2) == 1/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (ctx.int(1) / x.powi(2)).sqrt();
    assert_not_collapsed(&e, &(ctx.int(1) / &x), 0.5);
}

#[test]
fn pow_pow_x2_to_1_4_stays() {
    // sympy: simplify((x**2)**Rational(1,4)) == (x**2)**(1/4);
    //        N(((x**2)**Rational(1,4)).subs(x, -2), 17) == 1.4142135623730950
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(2).pow(&ctx.rational(1, 4));
    assert_not_collapsed(&e, &x.sqrt(), std::f64::consts::SQRT_2);
}

#[test]
fn pow_pow_cbrt_of_square_stays() {
    // sympy: simplify(cbrt(y**2)) == (y**2)**(1/3);
    //        N(cbrt(y**2).subs(y, -2), 17) == 1.5874010519681995
    let ctx = Context::new();
    let y = ctx.symbol("y");
    let e = y.powi(2).cbrt();
    assert_not_collapsed(&e, &y.pow(&ctx.rational(2, 3)), 1.5874010519681996);
}

#[test]
fn pow_pow_cbrt_of_cube_stays() {
    // sympy: simplify(cbrt(x**3)) == (x**3)**(1/3) — on the principal
    // branch cbrt(x**3).subs(x, -2) == 1 + sqrt(3)*I, not -2.  symplex's
    // eval_f64 takes the real cube root (-2 for both forms), so this case is
    // pinned structurally only.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(3).cbrt();
    let s = e.simplify();
    assert_ne!(s.id(), x.id(), "cbrt(x^3) must not collapse to x; got {s}");
    assert_eq!(
        s.id(),
        e.id(),
        "cbrt(x^3) should be left unchanged; got {s}"
    );
}

#[test]
fn pow_pow_cbrt_of_fourth_power_stays() {
    // sympy: simplify(cbrt(x**4)) == (x**4)**(1/3);
    //        N(cbrt(x**4).subs(x, -2), 17) == 2.5198420997897463
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(4).cbrt();
    assert_not_collapsed(&e, &x.pow(&ctx.rational(4, 3)), 2.5198420997897464);
}

#[test]
fn pow_pow_cbrt_of_shifted_square_stays() {
    // sympy: simplify(cbrt((x+1)**2)) == ((x + 1)**2)**(1/3);
    //        cbrt((x+1)**2).subs(x, -2) == 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let base = &x + ctx.int(1);
    let e = base.powi(2).cbrt();
    assert_not_collapsed(&e, &base.pow(&ctx.rational(2, 3)), 1.0);
}

#[test]
fn pow_pow_constant_cbrt_abs_sin6_squared_stays_real() {
    // sin(6) < 0, so sin(6)^(2/3) is complex on the principal branch while
    // cbrt(|sin(6)|²) is real.
    // sympy: N(cbrt(Abs(sin(6)**2)), 17) == 0.42739915660437737
    let ctx = Context::new();
    let e = ctx.int(6).sin().powi(2).abs().cbrt();
    let s = e.simplify();
    let v = s
        .eval_f64()
        .unwrap_or_else(|err| panic!("simplify({e}) = {s} is not real: {err}"));
    assert!(
        (v - 0.4273991566043774).abs() <= 1e-12,
        "simplify({e}) = {s} evaluates to {v}, sympy says 0.4273991566043774"
    );
}

// ── Cases that must still collapse ──────────────────────────────────────────

#[test]
fn pow_pow_sqrt_squared_collapses() {
    // sympy: (x**Rational(1,2))**2 == x  (integer outer exponent)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.pow(&ctx.rational(1, 2)).powi(2);
    assert_eq!(e.simplify().id(), x.id(), "(x^(1/2))^2 should be x");
}

#[test]
fn pow_pow_cbrt_cubed_collapses() {
    // sympy: (x**Rational(1,3))**3 == x  (integer outer exponent)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.pow(&ctx.rational(1, 3)).powi(3);
    assert_eq!(e.simplify().id(), x.id(), "(x^(1/3))^3 should be x");
}

#[test]
fn pow_pow_integer_integer_collapses() {
    // sympy: (x**2)**3 == x**6
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(2).powi(3);
    assert_eq!(e.simplify().id(), x.powi(6).id(), "(x^2)^3 should be x^6");
}

#[test]
fn pow_pow_inner_in_unit_interval_collapses() {
    // Inner exponent 1/2 ∈ (−1, 1]: arg(√x) ∈ (−π/2, π/2], so the principal
    // branches agree for every complex x.
    // sympy: (x**Rational(1,2))**Rational(1,2) == x**(1/4)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.pow(&ctx.rational(1, 2)).pow(&ctx.rational(1, 2));
    assert_eq!(
        e.simplify().id(),
        x.pow(&ctx.rational(1, 4)).id(),
        "(x^(1/2))^(1/2) should be x^(1/4)"
    );
}

// ── Assumptions ─────────────────────────────────────────────────────────────

#[test]
fn pow_pow_positive_base_collapses() {
    // sympy, xp = Symbol('xp', positive=True):
    //   (xp**2)**Rational(3,2) == xp**3; sqrt(1/xp**2) == 1/xp;
    //   (xp**2)**Rational(1,4) == sqrt(xp)
    let ctx = Context::new();
    let xp = ctx.symbol_with("xp", &[Assumption::Positive]).unwrap();
    let e = xp.powi(2).pow(&ctx.rational(3, 2));
    assert_eq!(
        e.simplify().id(),
        xp.powi(3).id(),
        "(xp^2)^(3/2) should be xp^3"
    );
    let e = (ctx.int(1) / xp.powi(2)).sqrt();
    assert_eq!(
        e.simplify().id(),
        (ctx.int(1) / &xp).id(),
        "sqrt(1/xp^2) should be 1/xp"
    );
    let e = xp.powi(2).pow(&ctx.rational(1, 4));
    assert_eq!(
        e.simplify().id(),
        xp.sqrt().id(),
        "(xp^2)^(1/4) should be sqrt(xp)"
    );
}

#[test]
fn pow_pow_negative_base_does_not_give_x_cubed() {
    // sympy, xn = Symbol('xn', negative=True): (xn**2)**Rational(3,2) == -xn**3,
    // and ((xn**2)**Rational(3,2)).subs(xn, -2) == 8.
    let ctx = Context::new();
    let xn = ctx.symbol_with("xn", &[Assumption::Negative]).unwrap();
    let e = xn.powi(2).pow(&ctx.rational(3, 2));
    let s = e.simplify();
    assert_ne!(
        s.id(),
        xn.powi(3).id(),
        "(xn^2)^(3/2) must not be xn^3 for xn < 0"
    );
    let v = s
        .subs(&xn, &ctx.int(-2))
        .eval_f64()
        .unwrap_or_else(|err| panic!("{s} at xn = -2 did not evaluate: {err}"));
    assert!(
        (v - 8.0).abs() <= 1e-12,
        "simplify({e}) = {s} at xn = -2 gives {v}, sympy says 8"
    );
}
