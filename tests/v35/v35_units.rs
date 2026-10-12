//! The units re-hunt (0.46): `ConstDim` arithmetic beyond the `i8`
//! exponent range, and exact conversion factors (`symplex::units`).

use symplex::prelude::*;
use symplex::units::*;

/// `ConstDim::mul` / `div` / `pow` saturate at the ends of the `i8`
/// range instead of overflowing.  Before, `ConstDim::new(100, …)
/// .mul(ConstDim::new(100, …))` panicked in a debug build ("attempt to
/// add with overflow") and gave the wrapped exponent `−56` in a release
/// build; `LENGTH.pow(100).pow(2)` gave `16` (wrapped `200`), and
/// `new(-100, …).div(new(100, …))` gave `56`.  Oracle: plain integer
/// arithmetic (`100 + 100 = 200 > 127`).
#[test]
fn const_dim_arithmetic_saturates_and_checked_variants_refuse() {
    let a = ConstDim::new(100, 0, -3, 0, 0, 0, 1);
    let b = ConstDim::new(100, 2, 0, 0, 0, 0, 0);
    let p = a.mul(b);
    assert_eq!((p.l, p.m, p.t, p.j), (127, 2, -3, 1));
    assert_eq!(a.checked_mul(b), None);
    let q = ConstDim::new(-100, 0, 0, 0, 0, 0, 0).div(b);
    assert_eq!((q.l, q.m), (-128, -2));
    assert_eq!(ConstDim::new(-100, 0, 0, 0, 0, 0, 0).checked_div(b), None);
    // −1 − (−128) = 127 fits: refused only when the true exponent leaves i8.
    assert_eq!(
        ConstDim::new(-1, 0, 0, 0, 0, 0, 0).checked_div(ConstDim::new(-128, 0, 0, 0, 0, 0, 0)),
        Some(ConstDim::new(127, 0, 0, 0, 0, 0, 0))
    );
    let big = ConstDim::LENGTH.pow(100);
    assert_eq!(big.pow(2).l, 127);
    assert_eq!(big.pow(-2).l, -128);
    assert_eq!(big.checked_pow(2), None);
    assert_eq!(ConstDim::new(-128, 0, 0, 0, 0, 0, 0).checked_pow(-1), None);
    // In range the checked forms agree with the plain ones.
    assert_eq!(
        ConstDim::MASS.checked_mul(ConstDim::ACCELERATION),
        Some(ConstDim::FORCE)
    );
    assert_eq!(
        ConstDim::ENERGY.checked_div(ConstDim::TIME),
        Some(ConstDim::POWER)
    );
    assert_eq!(
        ConstDim::LENGTH.checked_pow(-2),
        Some(ConstDim::new(-2, 0, 0, 0, 0, 0, 0))
    );
    assert_eq!(
        ConstDim::VOLTAGE.div(ConstDim::CURRENT),
        ConstDim::RESISTANCE
    );
}

/// Every conversion constructor, against the factor written from the
/// definitions (inch = 0.0254 m, pound = 0.45359237 kg, g_n = 9.80665 m/s²,
/// thermochemical calorie = 4.184 J, IT BTU = 1055.05585262 J, US gallon
/// = 231 in³, imperial gallon = 4.54609 L, atm = 101325 Pa, hp = 550
/// ft·lbf/s; NIST SP 811 appendix B).  The hunter found no wrong factor in
/// 0.45 (93 units); this pins the derived ones exactly.
#[test]
fn conversion_factors_match_the_definitions() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let check = |got: Ex, want: &str, label: &str| {
        let want = ctx.parse(want).unwrap();
        assert_eq!(
            (&got - &want).simplify().is_zero(),
            Some(true),
            "{label}: got {got}, want {want}"
        );
    };
    check(
        Length::miles(&one).into_inner(),
        "5280*12*254/10000",
        "mile",
    );
    check(
        Mass::slugs(&one).into_inner(),
        "45359237/100000000*980665/100000/(12*254/10000)",
        "slug",
    );
    check(
        Power::horsepower(&one).into_inner(),
        "550*12*254/10000*45359237/100000000*980665/100000",
        "hp",
    );
    check(
        Pressure::psi(&one).into_inner(),
        "45359237/100000000*980665/100000/(254/10000)^2",
        "psi",
    );
    check(
        Volume::us_tablespoons(&one).into_inner(),
        "231*(254/10000)^3/256",
        "tbsp",
    );
    check(
        Energy::btu(&one).into_inner(),
        "105505585262/100000000",
        "BTU",
    );
    check(
        Energy::foot_pounds(&one).into_inner(),
        "12*254/10000*45359237/100000000*980665/100000",
        "ft lbf",
    );
    check(AngularVelocity::rpm(&one).into_inner(), "2*pi/60", "rpm");
    check(
        Temperature::from_fahrenheit(&ctx.int(-40)).into_inner(),
        "4663/20",
        "-40 F",
    );
    check(
        Temperature::from_celsius(&ctx.int(-40)).into_inner(),
        "4663/20",
        "-40 C",
    );
}
