//! Tests of session 17, wave 16 (s15r1): powers of 0 with symbolic exponents, quotients by step functions.
//! Every test says what was wrong before and cites its oracle.
//!
//! The oracle of the values at points is exact arithmetic on the extended
//! complex numbers, as mpmath 1.3 and SymPy 1.14 compute it: `0^(13/10) = 0`
//! (`mpmath.power(0, 1.3)` → `0.0`), `0^(-13/10) = zoo` (`mpmath.power(0,
//! -1.3)` → `+inf`), `0^0 = 1`, `0^i = nan` (`mpmath.power(0, 1j)` → `nan`),
//! `0·zoo = nan`, `0/0 = nan`, `zoo·zoo = zoo`, `zoo − zoo = nan`
//! (`mpmath.inf - mpmath.inf` → `nan`), `0/zoo = 0`.

#![allow(unused_imports)]
use symplex::prelude::*;

/// `src` built, then `y` (and `z`) replaced at once and evaluated.
fn at(ctx: &Context, e: &Ex, y: &str, z: &str) -> String {
    let (sy, sz) = (ctx.symbol("y"), ctx.symbol("z"));
    let (vy, vz) = (ctx.parse(y).unwrap(), ctx.parse(z).unwrap());
    e.subs_map(&[(&sy, &vy), (&sz, &vz)]).eval().to_string()
}

/// `0^y·0^(−y)` was `1` and `0^y·0^z` was `0^(y + z)` up to 0.58: `canon_mul`
/// added the exponents of the base 0 like those of any other base.  For
/// `y = 13/10` the first is `0^(13/10)·0^(−13/10) = 0·zoo = nan` (oracle:
/// `mpmath.power(0, 1.3)*mpmath.power(0, -1.3)` → `nan`; SymPy
/// `S(0)**Rational(13,10)*S(0)**Rational(-13,10)` → `nan`), and `0^y·0^z` at
/// `(13/10, −13/10)` is `nan` where `0^(y + z) = 0^0 = 1`.
#[test]
fn exponents_of_the_base_zero_do_not_add() {
    let ctx = Context::new();
    let e = ctx.parse("0^y*0^(-y)").unwrap();
    assert_eq!(e.to_string(), "Piecewise(1 if y == 0, nan if True)");
    assert_eq!(at(&ctx, &e, "13/10", "0"), "nan");
    assert_eq!(at(&ctx, &e, "-13/10", "0"), "nan");
    assert_eq!(at(&ctx, &e, "0", "0"), "1");
    assert_eq!(at(&ctx, &e, "I", "0"), "nan");

    // Different exponents stay separate factors.
    let p = ctx.parse("0^y*0^z").unwrap();
    assert_eq!(p.to_string(), "0^y*0^z");
    assert_eq!(at(&ctx, &p, "13/10", "-13/10"), "nan"); // was 0^0 = 1
    assert_eq!(at(&ctx, &p, "13/10", "2"), "0");
    assert_eq!(at(&ctx, &p, "-13/10", "-1/2"), "zoo");
    assert_eq!(at(&ctx, &p, "0", "0"), "1");
    assert_eq!(at(&ctx, &p, "I", "2"), "nan");
    // ... also through the operations that merged them (was `0^(y + z)`).
    assert_eq!(p.simplify().to_string(), "0^y*0^z");
    assert_eq!(p.simplify_powers().to_string(), "0^y*0^z");
    assert_eq!(p.together().to_string(), "0^y*0^z");
    let q = ctx.parse("0^y/0^z").unwrap();
    assert_eq!(q.simplify().to_string(), "0^y*0^(-z)"); // was 0^(y - z)
    assert_eq!(at(&ctx, &q.simplify(), "13/10", "13/10"), "nan"); // was 0^0 = 1

    // Rational multiples of one exponent with one sign are of one kind at
    // every `y` (0, 1 or zoo together) and still add.
    assert_eq!(ctx.parse("0^y*0^y").unwrap().to_string(), "0^(2*y)");
    assert_eq!(ctx.parse("0^y*0^(2*y)").unwrap().to_string(), "0^(3*y)");
    assert_eq!(
        ctx.parse("0^(y/2)*0^(y/3)").unwrap().to_string(),
        "0^(5/6*y)"
    );
    // ... with both signs the product is 1 at 0 only (was 0^y).
    let r = ctx.parse("0^(2*y)*0^(-y)").unwrap();
    assert_eq!(r.to_string(), "Piecewise(1 if y == 0, nan if True)");
    // Another factor is kept beside the case distinction.
    let s = ctx.parse("x*0^y*0^(-y)*0^z").unwrap();
    assert_eq!(at(&ctx, &s, "0", "2"), "0");
    assert_eq!(at(&ctx, &s, "1", "2"), "nan"); // was x*0^z = 0
}

/// `(0^y)^q = 0^(q·y)` for a rational `q ≠ 0` at every `y` (each side is 0, 1,
/// `zoo` or `nan` together: `0^q = 0`, `zoo^q = zoo` for `q > 0`, swapped for
/// `q < 0`), so the powers of 0 have one form.  Up to 0.58 `0^y/0^y` stayed
/// `0^y·(0^y)⁻¹` when built and `simplify`, `ratsimp` and `powdenest` gave
/// `1`: for `y = 13/10` it is `0/0 = nan`, for `y = −13/10` `zoo/zoo = nan`
/// (oracle: SymPy `(S(0)**y/S(0)**y).subs(y, Rational(13, 10))` → `nan`).
#[test]
fn a_quotient_of_equal_powers_of_zero_is_not_one() {
    let ctx = Context::new();
    assert_eq!(ctx.parse("1/0^y").unwrap().to_string(), "0^(-y)");
    assert_eq!(ctx.parse("(0^y)^2").unwrap().to_string(), "0^(2*y)");
    assert_eq!(ctx.parse("sqrt(0^y)*0^y").unwrap().to_string(), "0^(3/2*y)");
    let e = ctx.parse("0^y/0^y").unwrap();
    for r in [
        e.clone(),
        e.simplify(),
        e.ratsimp(),
        e.powdenest(false),
        e.together(),
    ] {
        assert_eq!(r.to_string(), "Piecewise(1 if y == 0, nan if True)");
        assert_eq!(at(&ctx, &r, "13/10", "0"), "nan"); // was 1
        assert_eq!(at(&ctx, &r, "-13/10", "0"), "nan"); // was 1
        assert_eq!(at(&ctx, &r, "I", "0"), "nan"); // was 1
        assert_eq!(at(&ctx, &r, "0", "0"), "1");
    }
}

/// `zoo·0^y` was `zoo` up to 0.58 (`zoo` absorbed the power as a generic
/// nonzero factor).  For `re y > 0` it is `zoo·0 = nan` (oracle:
/// `mpmath.inf*mpmath.power(0, 1.3)` → `nan`; SymPy `zoo*S(0)**Rational(13,
/// 10)` → `nan`), for `re y < 0` `zoo·zoo = zoo`, at `y = 0` `zoo·1 = zoo`.
#[test]
fn zoo_times_a_power_of_zero() {
    let ctx = Context::new();
    let e = ctx.parse("zoo*0^y").unwrap();
    assert_eq!(
        e.to_string(),
        "Piecewise(zoo if 0 > re(y) | y == 0, nan if True)"
    );
    assert_eq!(at(&ctx, &e, "13/10", "0"), "nan"); // was zoo
    assert_eq!(at(&ctx, &e, "-13/10", "0"), "zoo");
    assert_eq!(at(&ctx, &e, "0", "0"), "zoo");
    assert_eq!(at(&ctx, &e, "I", "0"), "nan"); // was zoo
    // The reciprocal power: the other half plane.
    let r = ctx.parse("zoo/0^y").unwrap();
    assert_eq!(
        r.to_string(),
        "Piecewise(zoo if re(y) > 0 | y == 0, nan if True)"
    );
    assert_eq!(at(&ctx, &r, "-13/10", "0"), "nan"); // was zoo
    assert_eq!(at(&ctx, &r, "13/10", "0"), "zoo");
    // Both: `zoo` at 0 only.
    let b = ctx.parse("zoo*0^y*0^(-y)").unwrap();
    assert_eq!(at(&ctx, &b, "0", "0"), "zoo");
    assert_eq!(at(&ctx, &b, "13/10", "0"), "nan");
    assert_eq!(at(&ctx, &b, "-13/10", "0"), "nan");
    // A sum that is 0 or infinite on a region, as a factor and as a
    // denominator (`zoo/zoo` where it is infinite).
    let s = ctx.parse("zoo*(0^y - 0^z)").unwrap();
    assert_eq!(at(&ctx, &s, "13/10", "13/10"), "nan"); // zoo*0, was zoo
    assert_eq!(at(&ctx, &s, "13/10", "0"), "zoo");
    let d = ctx.parse("zoo/(0^y - 1)").unwrap();
    assert_eq!(at(&ctx, &d, "-13/10", "0"), "nan"); // zoo/zoo, was zoo
    assert_eq!(at(&ctx, &d, "13/10", "0"), "zoo");
    assert_eq!(at(&ctx, &d, "0", "0"), "zoo"); // zoo/0 = zoo*zoo
}

/// `0/(0^y − 0^z)` was `0` up to 0.58.  The denominator is 0 wherever the two
/// powers agree (`y, z > 0`: `0 − 0`; `y = z = 0`: `1 − 1`), where the
/// quotient is `0/0 = nan`; it has no value where both are `zoo` or one is
/// `nan` (oracle: SymPy `0/(S(0)**Rational(13,10) - S(0)**2)` → `nan`,
/// `0/(S(0)**Rational(13,10) - S(0)**Rational(-13,10))` → `0`).
#[test]
fn zero_over_a_difference_of_powers_of_zero() {
    let ctx = Context::new();
    let e = ctx.parse("0/(0^y - 0^z)").unwrap();
    assert_eq!(
        e.to_string(),
        "Piecewise(0 if abs(0^y - 0^z) > 0, nan if True)"
    );
    assert_eq!(at(&ctx, &e, "13/10", "2"), "nan"); // was 0
    assert_eq!(at(&ctx, &e, "0", "0"), "nan"); // was 0
    assert_eq!(at(&ctx, &e, "-13/10", "-1/2"), "nan"); // zoo - zoo, was 0
    assert_eq!(at(&ctx, &e, "I", "2"), "nan"); // was 0
    assert_eq!(at(&ctx, &e, "13/10", "-13/10"), "0");
    assert_eq!(at(&ctx, &e, "13/10", "0"), "0");
    // The single power keeps the conditions on its exponent (0.58).
    assert_eq!(
        ctx.parse("0/0^y").unwrap().to_string(),
        "Piecewise(0 if 0 > re(y) | y == 0, nan if True)"
    );
    assert_eq!(
        ctx.parse("0/(0^y*0^z)").unwrap().to_string(),
        "Piecewise(0 if (0 > re(y) | y == 0) & (0 > re(z) | z == 0), nan if True)"
    );
}

/// `0/H(y)`, `0/floor(y)`, `0/Max(y, 0)`, `0/KroneckerDelta(y, 1)` and
/// `0/DiracDelta(y)` were `0` up to 0.58: the generic convention (a
/// denominator with symbols is nonzero for almost every value of them) was
/// applied to functions that are 0 on a region.  Where the denominator is
/// 0 the quotient is `0/0 = nan` (oracle: SymPy `0/Heaviside(Rational(-13,
/// 10))` → `nan`, `0/floor(Rational(1, 2))` → `nan`, `0/Max(Rational(-13, 10),
/// 0)` → `nan`, `0/KroneckerDelta(Rational(13, 10), 1)` → `nan`,
/// `0/DiracDelta(Rational(13, 10))` → `nan`; with `Heaviside(0) = 1/2`).
#[test]
fn zero_over_a_step_function() {
    let ctx = Context::new();
    let h = ctx.parse("0/Heaviside(y)").unwrap();
    assert_eq!(h.to_string(), "Piecewise(0 if y >= 0, nan if True)");
    assert_eq!(at(&ctx, &h, "-13/10", "0"), "nan"); // was 0
    assert_eq!(at(&ctx, &h, "13/10", "0"), "0");
    assert_eq!(at(&ctx, &h, "0", "0"), "0"); // H(0) = 1/2

    let f = ctx.parse("0/floor(y)").unwrap();
    assert_eq!(
        f.to_string(),
        "Piecewise(0 if abs(floor(y)) > 0, nan if True)"
    );
    assert_eq!(at(&ctx, &f, "0", "0"), "nan"); // was 0
    assert_eq!(at(&ctx, &f, "1/2", "0"), "nan"); // was 0
    assert_eq!(at(&ctx, &f, "13/10", "0"), "0");
    assert_eq!(at(&ctx, &f, "-13/10", "0"), "0");

    let m = ctx.parse("0/Max(y, 0)").unwrap();
    assert_eq!(
        m.to_string(),
        "Piecewise(0 if abs(max(y, 0)) > 0, nan if True)"
    );
    assert_eq!(at(&ctx, &m, "-13/10", "0"), "nan"); // was 0
    assert_eq!(at(&ctx, &m, "0", "0"), "nan"); // was 0
    assert_eq!(at(&ctx, &m, "13/10", "0"), "0");
    // `max` has no value at a non-real argument (`Max(I, 0)` is `nan`): the
    // condition `|nan| > 0` has no truth value, and neither has the quotient.
    assert_eq!(at(&ctx, &m, "I", "0"), "nan"); // was 0

    let k = ctx.parse("0/KroneckerDelta(y, 1)").unwrap();
    assert_eq!(k.to_string(), "Piecewise(0 if y == 1, nan if True)");
    for p in ["13/10", "-13/10", "0", "I"] {
        assert_eq!(at(&ctx, &k, p, "0"), "nan"); // was 0
    }
    assert_eq!(at(&ctx, &k, "1", "0"), "0");

    let d = ctx.parse("0/DiracDelta(y)").unwrap();
    assert_eq!(d.to_string(), "Piecewise(0 if y == 0, nan if True)");
    for p in ["13/10", "-13/10", "I", "1"] {
        assert_eq!(at(&ctx, &d, p, "0"), "nan"); // was 0
    }

    // Through a product, a power, `|·|`; and `zoo` times a step (`zoo·0`).
    for src in [
        "0/(x*Heaviside(y))",
        "0/Heaviside(y)^2",
        "0/abs(Heaviside(y))",
    ] {
        assert_eq!(
            ctx.parse(src).unwrap().to_string(),
            "Piecewise(0 if y >= 0, nan if True)"
        );
    }
    let z = ctx.parse("zoo*Heaviside(y)").unwrap();
    assert_eq!(z.to_string(), "Piecewise(zoo if y >= 0, nan if True)");
    assert_eq!(at(&ctx, &z, "-13/10", "0"), "nan"); // was zoo
    assert_eq!(at(&ctx, &z, "13/10", "0"), "zoo");
    let zk = ctx.parse("zoo*KroneckerDelta(y, 1)").unwrap();
    assert_eq!(at(&ctx, &zk, "13/10", "0"), "nan"); // was zoo
    assert_eq!(at(&ctx, &zk, "1", "0"), "zoo");
}

/// The generic convention still holds for denominators that vanish at
/// isolated points only, and a sum the assumptions know nonzero needs no
/// condition: these were `0` and stay `0`.
#[test]
fn zero_over_a_generic_denominator_is_still_zero() {
    let ctx = Context::new();
    for src in [
        "0/y",
        "0/sin(y)",
        "0/abs(y)",
        "0/(y - 1)",
        "0/Max(y, 1)",
        "0/(KroneckerDelta(y, 1) + 1)",
        "0*Heaviside(y)",
        "0*floor(y)",
    ] {
        assert_eq!(ctx.parse(src).unwrap().to_string(), "0", "{src}");
    }
}

/// `limit(x/0^y, x, 0)` and `limit(x·0^y, x, 0)` were `0` up to 0.58: the
/// power was taken for a finite constant factor.  For `re y > 0` the first
/// is `x/0 = x·zoo`, `zoo` at every `x ≠ 0` (oracle: SymPy
/// `limit(x/S(0)**Rational(13, 10), x, 0)` → `zoo`; `mpmath.mpf('1e-30')
/// /mpmath.power(0, 1.3)` raises `ZeroDivisionError`), so the limit 0
/// holds for `re y < 0` (`x/zoo = 0`) and at `y = 0` (`x/1`) only.
#[test]
fn limit_of_a_vanishing_factor_times_a_power_of_zero() {
    let ctx = Context::new();
    let (x, zero) = (ctx.symbol("x"), ctx.zero());
    let e = ctx.parse("x/0^y").unwrap();
    let l = e.try_limit(&x, &zero).unwrap();
    assert_eq!(l.to_string(), "Piecewise(0 if 0 > re(y) | y == 0)");
    assert_eq!(at(&ctx, &l, "-13/10", "0"), "0");
    assert_eq!(at(&ctx, &l, "0", "0"), "0");
    assert_eq!(at(&ctx, &l, "13/10", "0"), "nan"); // was 0: no limit
    assert_eq!(at(&ctx, &l, "I", "0"), "nan"); // was 0: 0^(-i) has no value

    let m = ctx.parse("x*0^y").unwrap().try_limit(&x, &zero).unwrap();
    assert_eq!(m.to_string(), "Piecewise(0 if re(y) > 0 | y == 0)");
    assert_eq!(at(&ctx, &m, "-13/10", "0"), "nan"); // was 0: x*zoo
    assert_eq!(at(&ctx, &m, "13/10", "0"), "0");

    // In a sum, by direct substitution (was 1 at every y).
    let s = ctx
        .parse("x*0^y + 1")
        .unwrap()
        .try_limit(&x, &zero)
        .unwrap();
    assert_eq!(at(&ctx, &s, "13/10", "0"), "1");
    assert_eq!(at(&ctx, &s, "0", "0"), "1");
    assert_eq!(at(&ctx, &s, "-13/10", "0"), "nan"); // was 1

    // Beside a factor with the limit 1 the limit is the power itself where
    // it has a value (`zoo` is no limit value; refused up to 0.58).
    let c = ctx
        .parse("sin(x)*0^y/x")
        .unwrap()
        .try_limit(&x, &zero)
        .unwrap();
    assert_eq!(c.to_string(), "Piecewise(0 if re(y) > 0, 1 if y == 0)");
    // A generic limit that holds in every case stays: `x/(0^y + 1)` is
    // `x/1`, `x/2`, `x/zoo`.
    let g = ctx
        .parse("x/(0^y + 1)")
        .unwrap()
        .try_limit(&x, &zero)
        .unwrap();
    assert_eq!(g.to_string(), "0");
}

/// A `Piecewise` whose condition is an order relation with a `nan` operand
/// has no value: up to 0.58 `eval` kept `Piecewise((0, nan > 0), (nan,
/// True))` with the dead condition.  (`nan` has no order: IEEE 754, and
/// SymPy raises "Invalid NaN comparison".)
#[test]
fn a_piecewise_with_an_undefined_condition_has_no_value() {
    let ctx = Context::new();
    let (zero, nan, one) = (ctx.zero(), ctx.nan(), ctx.one());
    let p = Ex::piecewise(&[(&one, &nan.gt(&zero)), (&zero, &ctx.bool_true())]);
    assert_eq!(p.eval().to_string(), "nan");
    // An equality with `nan` is decided (false), as before.
    let q = Ex::piecewise(&[(&one, &nan.eq_expr(&zero)), (&zero, &ctx.bool_true())]);
    assert_eq!(q.eval().to_string(), "0");
}
