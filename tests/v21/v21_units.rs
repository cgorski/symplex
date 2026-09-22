//! 0.22 — every unit-conversion constructor in `units::conversions` pinned
//! to its defining factor, plus the never-called forwarding methods on
//! `Qty<D>` and the 30 named quantity newtypes, and the runtime helpers in
//! `units::dim` / `units::inference`.
//!
//! `conversions.rs` has no `to_<unit>()` getters: a constructor scales its
//! argument into SI and the value is read back with `inner()`, so each entry
//! pins (a) the exact rational the crate produces for one unit, (b) that
//! value against the *standard* definition as an `f64` at 1e-12, and (c)
//! linearity in a numeric and a symbolic argument.
//!
//! Factor sources (cited per table row; nothing below was computed by hand —
//! the rationals were reduced with sympy 1.14 `Rational` in
//! `symplex/.venv/bin/python`):
//!
//! * IYPA — International Yard and Pound Agreement (1959), reproduced in
//!   NIST SP 811 (2008) §B.6: 1 yd = 0.9144 m, 1 lb = 0.453 592 37 kg, exact.
//! * SP 811 — NIST Special Publication 811 (2008), Appendix B.8/B.9
//!   (conversion factors, "(exact)" marks).
//! * SI-9 — SI Brochure, 9th edition (BIPM 2019), Table 8 (min, h, d, °, ha,
//!   L, t, au, bar? — the non-SI units accepted for use with the SI).
//! * CGPM-3 (1901) g_n = 9.806 65 m/s²; CGPM-10 (1954) 1 atm = 101 325 Pa;
//!   Monaco 1929 hydrographic conference: 1 nautical mile = 1852 m.
//! * ISO 80000-5:2019 Annex: cal_th = 4.184 J; BTU_IT = 1055.055 852 62 J.
//! * UK Weights and Measures Act 1985, Sch. 1: 1 gallon = 4.546 09 L.

use std::f64::consts::{PI, TAU};

use symplex::num_bigint::BigInt;
use symplex::num_rational::Ratio;
use symplex::prelude::*;
use symplex::units::conv_factors::{base, derived};
use symplex::units::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// `q` folded to a literal must be exactly `num/den` (any representative).
fn exact(q: &impl AsRef<Ex>, num: i64, den: i64, label: &str) {
    let ex = q.as_ref().eval();
    let got = ex
        .as_rational()
        .unwrap_or_else(|| panic!("{label}: `{ex}` is not a rational literal"));
    let want = Ratio::new(BigInt::from(num), BigInt::from(den));
    assert_eq!(got, want, "{label}: got {got}, want {num}/{den}");
}

/// Relative closeness at 1e-12 (the "against the standard definition" check).
fn close(actual: f64, expected: f64, label: &str) {
    let scale = expected.abs().max(f64::MIN_POSITIVE);
    assert!(
        (actual - expected).abs() <= 1e-12 * scale,
        "{label}: got {actual:e}, expected {expected:e} (relative error {:e})",
        (actual - expected).abs() / scale
    );
}

/// One linear constructor: `name`, the constructor, the exact SI factor
/// `num/den` the crate must produce for one unit, and the standard
/// definition as an `f64`.
type Row<T> = (&'static str, fn(&Ex) -> T, (i64, i64), f64);

/// (a) exact factor, (b) f64 against the definition, (c) linearity in a
/// numeric (7) and a symbolic (`x`) argument.
fn check_linear<T: AsRef<Ex>>(ctx: &Context, rows: &[Row<T>]) {
    let one = ctx.int(1);
    let seven = ctx.int(7);
    let x = ctx.symbol("x");
    for &(name, ctor, (num, den), def) in rows {
        exact(&ctor(&one), num, den, name);
        close(
            ctor(&one).as_ref().eval_f64().expect(name),
            def,
            &format!("{name} vs definition"),
        );
        exact(&ctor(&seven), 7 * num, den, &format!("7 {name}"));
        let want = &x * &ctx.rational(num, den);
        assert_eq!(
            ctor(&x).as_ref().equals(&want),
            Some(true),
            "{name}: symbolic argument should scale to `{want}`, got `{}`",
            ctor(&x).as_ref()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// conv_factors — the crate's own constants against their definitions
// ═══════════════════════════════════════════════════════════════════════════

/// The exact base factors of `conv_factors::base` (SP 811 §B.6/B.8, CGPM,
/// ISO 80000-5).  The astronomical unit (IAU 2012 Resolution B2,
/// 149 597 870 700 m) is defined here but no constructor uses it.
#[test]
fn conv_factor_base_constants_match_their_definitions() {
    assert_eq!(base::INCH, (127, 5000)); // IYPA: 1 in = 0.0254 m
    assert_eq!(base::POUND, (45_359_237, 100_000_000)); // IYPA
    assert_eq!(base::G_N, (980_665, 100_000)); // CGPM-3 1901
    assert_eq!(base::CALORIE_TH, (4184, 1000)); // ISO 80000-5
    assert_eq!(base::ATM, (101_325, 1)); // CGPM-10 1954
    assert_eq!(base::NAUTICAL_MILE, (1852, 1)); // Monaco 1929
    assert_eq!(base::ASTRONOMICAL_UNIT, (149_597_870_700, 1)); // IAU 2012 B2
    assert_eq!(base::BTU_IT, (52_752_792_631, 50_000_000)); // 1055.05585262 J
    assert_eq!(base::IMPERIAL_GALLON, (454_609, 100_000_000)); // 4.54609 L
}

/// Every `conv_factors::derived` constant recomputed from the base ones in
/// `Ratio<BigInt>` (sympy: e.g. `33000*12*Rational(254,10000)*lb*g_n/60`
/// = 37284993579113511/50000000000000).
#[test]
fn conv_factor_derived_constants_recompute_from_base() {
    let r = |(n, d): (i64, i64)| Ratio::new(BigInt::from(n), BigInt::from(d));
    let inch = r(base::INCH);
    let lb = r(base::POUND);
    let g_n = r(base::G_N);
    let atm = r(base::ATM);
    let nmi = r(base::NAUTICAL_MILE);
    let ft = &inch * BigInt::from(12);
    let yd = &ft * BigInt::from(3);
    let mi = &ft * BigInt::from(5280);
    let lbf = &lb * &g_n;
    let gal = &inch * &inch * &inch * BigInt::from(231);
    let cases: [(&str, Ratio<BigInt>, (i64, i64)); 24] = [
        ("FOOT", ft.clone(), derived::FOOT),
        ("YARD", yd.clone(), derived::YARD),
        ("MILE", mi.clone(), derived::MILE),
        ("FATHOM", &yd * BigInt::from(2), derived::FATHOM),
        ("POUND_FORCE", lbf.clone(), derived::POUND_FORCE),
        ("KILOGRAM_FORCE", g_n.clone(), derived::KILOGRAM_FORCE),
        (
            "HORSEPOWER",
            &ft * &lbf * BigInt::from(33_000) / BigInt::from(60),
            derived::HORSEPOWER,
        ),
        (
            "METRIC_HORSEPOWER",
            &g_n * BigInt::from(75),
            derived::METRIC_HORSEPOWER,
        ),
        ("PSI", &lbf / (&inch * &inch), derived::PSI),
        ("TORR", &atm / BigInt::from(760), derived::TORR),
        ("FOOT_POUND", &ft * &lbf, derived::FOOT_POUND),
        ("US_GALLON", gal.clone(), derived::US_GALLON),
        ("US_QUART", &gal / BigInt::from(4), derived::US_QUART),
        ("US_PINT", &gal / BigInt::from(8), derived::US_PINT),
        (
            "US_FLUID_OUNCE",
            &gal / BigInt::from(128),
            derived::US_FLUID_OUNCE,
        ),
        (
            "US_TABLESPOON",
            &gal / BigInt::from(256),
            derived::US_TABLESPOON,
        ),
        ("OUNCE", &lb / BigInt::from(16), derived::OUNCE),
        ("SHORT_TON", &lb * BigInt::from(2000), derived::SHORT_TON),
        ("LONG_TON", &lb * BigInt::from(2240), derived::LONG_TON),
        ("GRAIN", &lb / BigInt::from(7000), derived::GRAIN),
        ("SLUG", &lbf / &ft, derived::SLUG),
        ("KNOT", &nmi / BigInt::from(3600), derived::KNOT),
        ("MPH", &mi / BigInt::from(3600), derived::MPH),
        ("DYNE", r((1, 100_000)), derived::DYNE),
    ];
    for (name, computed, stored) in cases {
        assert_eq!(computed, r(stored), "{name}");
    }
    assert_eq!(derived::ERG, (1, 10_000_000));
}

// ═══════════════════════════════════════════════════════════════════════════
// Length [m]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn length_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Length>(
        &ctx,
        &[
            ("meters", Length::meters, (1, 1), 1.0),
            ("kilometers", Length::kilometers, (1000, 1), 1e3),
            ("centimeters", Length::centimeters, (1, 100), 1e-2),
            ("millimeters", Length::millimeters, (1, 1000), 1e-3),
            ("micrometers", Length::micrometers, (1, 1_000_000), 1e-6),
            // IYPA / SP 811 B.8: 1 in = 2.54 cm (exact)
            ("inches", Length::inches, (127, 5000), 0.0254),
            // SP 811 B.8: 1 ft = 0.3048 m (exact)
            ("feet", Length::feet, (381, 1250), 0.3048),
            // IYPA: 1 yd = 0.9144 m (exact)
            ("yards", Length::yards, (1143, 1250), 0.9144),
            // SP 811 B.8: 1 mi (international) = 1609.344 m (exact)
            ("miles", Length::miles, (201_168, 125), 1609.344),
            // Monaco 1929 / SP 811 B.8: 1 nmi = 1852 m (exact)
            ("nautical_miles", Length::nautical_miles, (1852, 1), 1852.0),
            // 1 fathom = 2 yd = 1.8288 m on the international yard; SP 811 B.8
            // lists the *US survey* fathom 1.828 804 m (differs by 2 ppm) —
            // the crate documents the international definition.
            ("fathoms", Length::fathoms, (1143, 625), 1.8288),
            // SP 811 B.8: 1 mil = 0.001 in = 2.54e-5 m (exact)
            ("mils", Length::mils, (127, 5_000_000), 2.54e-5),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Mass [kg]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mass_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Mass>(
        &ctx,
        &[
            ("kilograms", Mass::kilograms, (1, 1), 1.0),
            ("grams", Mass::grams, (1, 1000), 1e-3),
            ("milligrams", Mass::milligrams, (1, 1_000_000), 1e-6),
            // SI-9 Table 8: 1 t = 10³ kg
            ("tonnes", Mass::tonnes, (1000, 1), 1e3),
            // IYPA: 1 lb = 0.453 592 37 kg (exact)
            (
                "pounds",
                Mass::pounds,
                (45_359_237, 100_000_000),
                0.45359237,
            ),
            // SP 811 B.8: 1 oz (avdp) = lb/16 = 2.834 952 3125e-2 kg (exact)
            (
                "ounces",
                Mass::ounces,
                (45_359_237, 1_600_000_000),
                0.028349523125,
            ),
            // SP 811 B.8: 1 short ton = 2000 lb = 907.184 74 kg (exact)
            (
                "short_tons",
                Mass::short_tons,
                (45_359_237, 50_000),
                907.18474,
            ),
            // SP 811 B.8: 1 long ton = 2240 lb = 1016.046 9088 kg (exact)
            (
                "long_tons",
                Mass::long_tons,
                (317_514_659, 312_500),
                1016.0469088,
            ),
            // SP 811 B.8: 1 grain = lb/7000 = 6.479 891e-5 kg (exact)
            (
                "grains",
                Mass::grains,
                (6_479_891, 100_000_000_000),
                6.479891e-5,
            ),
            // SP 811 B.8: 1 slug = 1 lbf·s²/ft = 14.593 902 937 206 36 kg
            (
                "slugs",
                Mass::slugs,
                (8_896_443_230_521, 609_600_000_000),
                14.593902937206364,
            ),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Time [s]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn time_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Time>(
        &ctx,
        &[
            ("seconds", Time::seconds, (1, 1), 1.0),
            ("milliseconds", Time::milliseconds, (1, 1000), 1e-3),
            ("microseconds", Time::microseconds, (1, 1_000_000), 1e-6),
            ("nanoseconds", Time::nanoseconds, (1, 1_000_000_000), 1e-9),
            // SI-9 Table 8: min = 60 s, h = 3600 s, d = 86 400 s
            ("minutes", Time::minutes, (60, 1), 60.0),
            ("hours", Time::hours, (3600, 1), 3600.0),
            ("days", Time::days, (86_400, 1), 86_400.0),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Angle [rad] — π-valued factors are checked symbolically
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn angle_radians_is_identity() {
    let ctx = Context::new();
    check_linear::<Angle>(&ctx, &[("radians", Angle::radians, (1, 1), 1.0)]);
}

/// SI-9 Table 8: 1° = (π/180) rad.
#[test]
fn angle_degrees_is_pi_over_180() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let want = &ctx.pi() / 180;
    assert_eq!(Angle::degrees(&one).inner().equals(&want), Some(true));
    close(Angle::degrees(&one).eval_f64().unwrap(), PI / 180.0, "1°");
    // 180° = π rad exactly.
    assert_eq!(
        Angle::degrees(&ctx.int(180)).inner().equals(&ctx.pi()),
        Some(true)
    );
}

/// 1 revolution = 2π rad.
#[test]
fn angle_revolutions_is_two_pi() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let want = 2 * &ctx.pi();
    assert_eq!(Angle::revolutions(&one).inner().equals(&want), Some(true));
    close(Angle::revolutions(&one).eval_f64().unwrap(), TAU, "1 rev");
    // Half a revolution is π.
    let half = ctx.rational(1, 2);
    assert_eq!(
        Angle::revolutions(&half).inner().equals(&ctx.pi()),
        Some(true)
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Velocity [m/s]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn velocity_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Velocity>(
        &ctx,
        &[
            (
                "meters_per_second",
                Velocity::meters_per_second,
                (1, 1),
                1.0,
            ),
            // 1 km/h = 1000/3600 m/s = 5/18 m/s (SP 811 B.8: 2.777 778e-1)
            (
                "kilometers_per_hour",
                Velocity::kilometers_per_hour,
                (5, 18),
                5.0 / 18.0,
            ),
            // SP 811 B.8: 1 mi/h = 0.447 04 m/s (exact)
            (
                "miles_per_hour",
                Velocity::miles_per_hour,
                (1397, 3125),
                0.44704,
            ),
            // SP 811 B.8: 1 knot = 1852/3600 m/s = 0.514 444 … m/s
            ("knots", Velocity::knots, (463, 900), 463.0 / 900.0),
            // SP 811 B.8: 1 ft/s = 0.3048 m/s (exact)
            (
                "feet_per_second",
                Velocity::feet_per_second,
                (381, 1250),
                0.3048,
            ),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Force [N]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn force_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Force>(
        &ctx,
        &[
            ("newtons", Force::newtons, (1, 1), 1.0),
            ("kilonewtons", Force::kilonewtons, (1000, 1), 1e3),
            // SP 811 B.8: 1 lbf = 4.448 221 615 260 5 N (exact: lb × g_n)
            (
                "pound_force",
                Force::pound_force,
                (8_896_443_230_521, 2_000_000_000_000),
                4.4482216152605,
            ),
            // SP 811 B.8: 1 kgf = 9.806 65 N (exact)
            (
                "kilogram_force",
                Force::kilogram_force,
                (196_133, 20_000),
                9.80665,
            ),
            // SP 811 B.8: 1 dyn = 1e-5 N (exact)
            ("dynes", Force::dynes, (1, 100_000), 1e-5),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Energy [J]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn energy_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Energy>(
        &ctx,
        &[
            ("joules", Energy::joules, (1, 1), 1.0),
            ("kilojoules", Energy::kilojoules, (1000, 1), 1e3),
            // SP 811 B.8: 1 kW·h = 3.6e6 J (exact)
            (
                "kilowatt_hours",
                Energy::kilowatt_hours,
                (3_600_000, 1),
                3.6e6,
            ),
            // ISO 80000-5 / SP 811 B.8: 1 cal_th = 4.184 J (exact)
            ("calories", Energy::calories, (523, 125), 4.184),
            ("kilocalories", Energy::kilocalories, (4184, 1), 4184.0),
            // SP 811 B.8: 1 Btu_IT = 1.055 055 852 62e3 J (exact)
            (
                "btu",
                Energy::btu,
                (52_752_792_631, 50_000_000),
                1055.05585262,
            ),
            // SP 811 B.8: 1 erg = 1e-7 J (exact)
            ("ergs", Energy::ergs, (1, 10_000_000), 1e-7),
            // SP 811 B.8: 1 ft·lbf = 1.355 817 948 331 400 4 J (ft × lbf, exact)
            (
                "foot_pounds",
                Energy::foot_pounds,
                (3_389_544_870_828_501, 2_500_000_000_000_000),
                1.3558179483314003,
            ),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Power [W]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn power_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Power>(
        &ctx,
        &[
            ("watts", Power::watts, (1, 1), 1.0),
            ("kilowatts", Power::kilowatts, (1000, 1), 1e3),
            ("megawatts", Power::megawatts, (1_000_000, 1), 1e6),
            // SP 811 B.8: 1 hp (550 ft·lbf/s) = 745.699 871 582 270 22 W
            (
                "horsepower",
                Power::horsepower,
                (37_284_993_579_113_511, 50_000_000_000_000),
                745.6998715822702,
            ),
            // SP 811 B.8: 1 hp (metric) = 75 kgf·m/s = 735.498 75 W (exact)
            (
                "metric_horsepower",
                Power::metric_horsepower,
                (588_399, 800),
                735.49875,
            ),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Electrical: Voltage [V], Current [A], Resistance [Ω]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn voltage_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Voltage>(
        &ctx,
        &[
            ("volts", Voltage::volts, (1, 1), 1.0),
            ("millivolts", Voltage::millivolts, (1, 1000), 1e-3),
            ("kilovolts", Voltage::kilovolts, (1000, 1), 1e3),
        ],
    );
}

#[test]
fn current_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Current>(
        &ctx,
        &[
            ("amperes", Current::amperes, (1, 1), 1.0),
            ("milliamperes", Current::milliamperes, (1, 1000), 1e-3),
            ("microamperes", Current::microamperes, (1, 1_000_000), 1e-6),
        ],
    );
}

#[test]
fn resistance_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Resistance>(
        &ctx,
        &[
            ("ohms", Resistance::ohms, (1, 1), 1.0),
            ("milliohms", Resistance::milliohms, (1, 1000), 1e-3),
            ("kilohms", Resistance::kilohms, (1000, 1), 1e3),
            ("megohms", Resistance::megohms, (1_000_000, 1), 1e6),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Pressure [Pa]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pressure_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Pressure>(
        &ctx,
        &[
            ("pascals", Pressure::pascals, (1, 1), 1.0),
            ("kilopascals", Pressure::kilopascals, (1000, 1), 1e3),
            ("megapascals", Pressure::megapascals, (1_000_000, 1), 1e6),
            // SI-9 Table 8: 1 bar = 10⁵ Pa
            ("bars", Pressure::bars, (100_000, 1), 1e5),
            // CGPM-10 1954: 1 atm = 101 325 Pa (exact)
            (
                "atmospheres",
                Pressure::atmospheres,
                (101_325, 1),
                101_325.0,
            ),
            // SP 811 B.8: 1 psi = lbf/in² = 6894.757 293 168 362 Pa
            (
                "psi",
                Pressure::psi,
                (8_896_443_230_521, 1_290_320_000),
                6894.757293168362,
            ),
            // SP 811 B.8: 1 torr = 101 325/760 Pa = 133.322 368 421 05 Pa
            // (NB: the conventional mmHg, 133.322 387 415 Pa, is a different
            // unit; the crate exposes torr only.)
            ("torr", Pressure::torr, (20_265, 152), 133.32236842105263),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Frequency [Hz]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn frequency_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Frequency>(
        &ctx,
        &[
            ("hertz", Frequency::hertz, (1, 1), 1.0),
            ("kilohertz", Frequency::kilohertz, (1000, 1), 1e3),
            ("megahertz", Frequency::megahertz, (1_000_000, 1), 1e6),
            ("gigahertz", Frequency::gigahertz, (1_000_000_000, 1), 1e9),
            // 1 r/min = 1/60 Hz (SP 811 B.8: 1.666 667e-2 Hz)
            ("rpm", Frequency::rpm, (1, 60), 1.0 / 60.0),
            ("bpm", Frequency::bpm, (1, 60), 1.0 / 60.0),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Temperature [K] — affine scales
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn temperature_kelvins_is_identity_and_rankine_is_five_ninths() {
    let ctx = Context::new();
    check_linear::<Temperature>(
        &ctx,
        &[
            ("kelvins", Temperature::kelvins, (1, 1), 1.0),
            // SP 811 B.8: T/K = T/°R × 5/9
            ("from_rankine", Temperature::from_rankine, (5, 9), 5.0 / 9.0),
        ],
    );
}

/// SI-9 §2.3.1 footnote / SP 811 B.8: T/K = t/°C + 273.15.
#[test]
fn temperature_from_celsius_offsets_by_273_15() {
    let ctx = Context::new();
    exact(&Temperature::from_celsius(&ctx.int(0)), 5463, 20, "0 °C");
    exact(
        &Temperature::from_celsius(&ctx.int(100)),
        7463,
        20,
        "100 °C",
    );
    exact(
        &Temperature::from_celsius(&ctx.rational(-27_315, 100)),
        0,
        1,
        "−273.15 °C",
    );
}

/// SP 811 B.8: T/K = (t/°F + 459.67) / 1.8.
#[test]
fn temperature_from_fahrenheit_is_affine_five_ninths() {
    let ctx = Context::new();
    exact(
        &Temperature::from_fahrenheit(&ctx.int(32)),
        5463,
        20,
        "32 °F",
    );
    exact(
        &Temperature::from_fahrenheit(&ctx.int(212)),
        7463,
        20,
        "212 °F",
    );
    exact(
        &Temperature::from_fahrenheit(&ctx.int(0)),
        45_967,
        180,
        "0 °F",
    );
    close(
        Temperature::from_fahrenheit(&ctx.int(0))
            .eval_f64()
            .unwrap(),
        255.37222222222223,
        "0 °F",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Torque, Momentum, MagneticFlux, Damping — SI-only identity constructors
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn torque_newton_meters_is_identity() {
    let ctx = Context::new();
    check_linear::<Torque>(
        &ctx,
        &[("newton_meters", Torque::newton_meters, (1, 1), 1.0)],
    );
}

#[test]
fn momentum_kilogram_meters_per_second_is_identity() {
    let ctx = Context::new();
    check_linear::<Momentum>(
        &ctx,
        &[(
            "kilogram_meters_per_second",
            Momentum::kilogram_meters_per_second,
            (1, 1),
            1.0,
        )],
    );
}

#[test]
fn magnetic_flux_webers_is_identity() {
    let ctx = Context::new();
    check_linear::<MagneticFlux>(&ctx, &[("webers", MagneticFlux::webers, (1, 1), 1.0)]);
}

#[test]
fn damping_newton_seconds_per_meter_is_identity() {
    let ctx = Context::new();
    check_linear::<Damping>(
        &ctx,
        &[(
            "newton_seconds_per_meter",
            Damping::newton_seconds_per_meter,
            (1, 1),
            1.0,
        )],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Acceleration [m/s²]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn acceleration_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Acceleration>(
        &ctx,
        &[
            (
                "meters_per_second_squared",
                Acceleration::meters_per_second_squared,
                (1, 1),
                1.0,
            ),
            // CGPM-3 1901: g_n = 9.806 65 m/s² (exact)
            (
                "standard_gravity",
                Acceleration::standard_gravity,
                (196_133, 20_000),
                9.80665,
            ),
            // SP 811 B.8: 1 ft/s² = 0.3048 m/s² (exact)
            (
                "feet_per_second_squared",
                Acceleration::feet_per_second_squared,
                (381, 1250),
                0.3048,
            ),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Area [m²]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn area_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Area>(
        &ctx,
        &[
            ("square_meters", Area::square_meters, (1, 1), 1.0),
            (
                "square_kilometers",
                Area::square_kilometers,
                (1_000_000, 1),
                1e6,
            ),
            (
                "square_centimeters",
                Area::square_centimeters,
                (1, 10_000),
                1e-4,
            ),
            // SI-9 Table 8: 1 ha = 10⁴ m²
            ("hectares", Area::hectares, (10_000, 1), 1e4),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Volume [m³]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn volume_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Volume>(
        &ctx,
        &[
            ("cubic_meters", Volume::cubic_meters, (1, 1), 1.0),
            // SI-9 Table 8: 1 L = 10⁻³ m³
            ("liters", Volume::liters, (1, 1000), 1e-3),
            ("milliliters", Volume::milliliters, (1, 1_000_000), 1e-6),
            // SP 811 B.8: 1 gal (US) = 231 in³ = 3.785 411 784e-3 m³ (exact)
            (
                "us_gallons",
                Volume::us_gallons,
                (473_176_473, 125_000_000_000),
                3.785411784e-3,
            ),
            // SP 811 B.8: 1 qt (US liquid) = gal/4 = 9.463 529 46e-4 m³ (exact)
            (
                "us_quarts",
                Volume::us_quarts,
                (473_176_473, 500_000_000_000),
                9.46352946e-4,
            ),
            // SP 811 B.8: 1 pt (US liquid) = gal/8 = 4.731 764 73e-4 m³ (exact)
            (
                "us_pints",
                Volume::us_pints,
                (473_176_473, 1_000_000_000_000),
                4.73176473e-4,
            ),
            // SP 811 B.8: 1 fl oz (US) = gal/128 = 2.957 352 956 25e-5 m³ (exact)
            (
                "us_fluid_ounces",
                Volume::us_fluid_ounces,
                (473_176_473, 16_000_000_000_000),
                2.95735295625e-5,
            ),
            // UK W&M Act 1985: 1 gal (UK) = 4.546 09 L (exact)
            (
                "imperial_gallons",
                Volume::imperial_gallons,
                (454_609, 100_000_000),
                4.54609e-3,
            ),
            // SP 811 B.8: 1 tablespoon = fl oz/2 = 1.478 676 478 125e-5 m³ (exact)
            (
                "us_tablespoons",
                Volume::us_tablespoons,
                (473_176_473, 32_000_000_000_000),
                1.478676478125e-5,
            ),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// AngularVelocity [rad/s]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn angular_velocity_radians_per_second_is_identity() {
    let ctx = Context::new();
    check_linear::<AngularVelocity>(
        &ctx,
        &[(
            "radians_per_second",
            AngularVelocity::radians_per_second,
            (1, 1),
            1.0,
        )],
    );
}

/// 1 r/min = 2π/60 rad/s = π/30 rad/s.
#[test]
fn angular_velocity_rpm_is_pi_over_30() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let want = &ctx.pi() / 30;
    assert_eq!(AngularVelocity::rpm(&one).inner().equals(&want), Some(true));
    close(
        AngularVelocity::rpm(&one).eval_f64().unwrap(),
        PI / 30.0,
        "1 rpm",
    );
    // 60 rpm = 2π rad/s.
    assert_eq!(
        AngularVelocity::rpm(&ctx.int(60))
            .inner()
            .equals(&(2 * &ctx.pi())),
        Some(true)
    );
}

/// 1 °/s = π/180 rad/s.
#[test]
fn angular_velocity_degrees_per_second_is_pi_over_180() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let want = &ctx.pi() / 180;
    assert_eq!(
        AngularVelocity::degrees_per_second(&one)
            .inner()
            .equals(&want),
        Some(true)
    );
    close(
        AngularVelocity::degrees_per_second(&one)
            .eval_f64()
            .unwrap(),
        PI / 180.0,
        "1 °/s",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Charge [C], Capacitance [F], Inductance [H]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn charge_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Charge>(
        &ctx,
        &[
            ("coulombs", Charge::coulombs, (1, 1), 1.0),
            // 1 mA·h = 1e-3 A × 3600 s = 3.6 C (exact)
            ("milliampere_hours", Charge::milliampere_hours, (18, 5), 3.6),
            // SP 811 B.8: 1 A·h = 3600 C (exact)
            ("ampere_hours", Charge::ampere_hours, (3600, 1), 3600.0),
        ],
    );
}

#[test]
fn capacitance_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Capacitance>(
        &ctx,
        &[
            ("farads", Capacitance::farads, (1, 1), 1.0),
            (
                "microfarads",
                Capacitance::microfarads,
                (1, 1_000_000),
                1e-6,
            ),
            (
                "nanofarads",
                Capacitance::nanofarads,
                (1, 1_000_000_000),
                1e-9,
            ),
            (
                "picofarads",
                Capacitance::picofarads,
                (1, 1_000_000_000_000),
                1e-12,
            ),
        ],
    );
}

#[test]
fn inductance_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Inductance>(
        &ctx,
        &[
            ("henrys", Inductance::henrys, (1, 1), 1.0),
            ("millihenrys", Inductance::millihenrys, (1, 1000), 1e-3),
            ("microhenrys", Inductance::microhenrys, (1, 1_000_000), 1e-6),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Stiffness [N/m], Dimensionless
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn stiffness_constructors_pin_their_factors() {
    let ctx = Context::new();
    check_linear::<Stiffness>(
        &ctx,
        &[
            (
                "newtons_per_meter",
                Stiffness::newtons_per_meter,
                (1, 1),
                1.0,
            ),
            (
                "kilonewtons_per_meter",
                Stiffness::kilonewtons_per_meter,
                (1000, 1),
                1e3,
            ),
        ],
    );
}

#[test]
fn dimensionless_percent_and_per_mille() {
    let ctx = Context::new();
    check_linear::<Dimensionless>(
        &ctx,
        &[
            ("percent", Dimensionless::percent, (1, 100), 1e-2),
            ("per_mille", Dimensionless::per_mille, (1, 1000), 1e-3),
        ],
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-unit identities — the factors must compose
// ═══════════════════════════════════════════════════════════════════════════

/// 1 psi × 1 in² = 1 lbf; 1 ft·lbf = 1 ft × 1 lbf; 1 hp = 550 ft·lbf/s;
/// 1 slug × 1 ft/s² = 1 lbf — all exact in the crate's rationals.
#[test]
fn imperial_factors_compose_exactly() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let inch = Length::inches(&one).into_inner();
    let psi = Pressure::psi(&one).into_inner();
    let lbf = Force::pound_force(&one).into_inner();
    let ft = Length::feet(&one).into_inner();
    let ftlbf = Energy::foot_pounds(&one).into_inner();
    let hp = Power::horsepower(&one).into_inner();
    let slug = Mass::slugs(&one).into_inner();
    let ft_s2 = Acceleration::feet_per_second_squared(&one).into_inner();
    assert_eq!((&psi * &inch * &inch).eval().equals(&lbf), Some(true));
    assert_eq!((&ft * &lbf).eval().equals(&ftlbf), Some(true));
    assert_eq!((&ftlbf * 550).eval().equals(&hp), Some(true));
    assert_eq!((&slug * &ft_s2).eval().equals(&lbf), Some(true));
}

/// 1 US gallon = 4 qt = 8 pt = 128 fl oz = 256 tbsp = 231 in³, exact.
#[test]
fn us_customary_volumes_compose_exactly() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let gal = Volume::us_gallons(&one).into_inner();
    let inch = Length::inches(&one).into_inner();
    for (n, ex) in [
        (4, Volume::us_quarts(&one)),
        (8, Volume::us_pints(&one)),
        (128, Volume::us_fluid_ounces(&one)),
        (256, Volume::us_tablespoons(&one)),
    ] {
        assert_eq!(
            (ex.inner() * n).eval().equals(&gal),
            Some(true),
            "{n} × {ex:?} = 1 gal"
        );
    }
    assert_eq!(
        (inch.powi(3) * 231).eval().equals(&gal),
        Some(true),
        "231 in³ = 1 gal"
    );
    // 1 imperial gallon = 4.54609 L = 1.200 949 925 504 855 US gallons
    // (sympy: float(Rational(454609, 10**8) / (231 * Rational(254, 10**4)**3))).
    close(
        Volume::imperial_gallons(&one).eval_f64().unwrap() / gal.eval_f64().unwrap(),
        1.200949925504855,
        "UK gal / US gal",
    );
}

/// 1 kcal = 1000 cal_th, 1 Btu_IT = 1055.05585262/4.184 cal_th, 1 kW·h =
/// 3.6 MJ, 1 hp·h = 2 684 519.537 696 172 79 J (SP 811 B.8: 2.684 520e6).
#[test]
fn energy_factors_compose_exactly() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let cal = Energy::calories(&one).into_inner();
    let kcal = Energy::kilocalories(&one).into_inner();
    assert_eq!((&cal * 1000).eval().equals(&kcal), Some(true));
    let kwh = Energy::kilowatt_hours(&one).into_inner();
    let kw_times_h = Power::kilowatts(&one).into_inner() * Time::hours(&one).into_inner();
    assert_eq!(kw_times_h.eval().equals(&kwh), Some(true));
    // sympy: Rational(37284993579113511, 50000000000000) * 3600
    let hp_h = Power::horsepower(&one).into_inner() * Time::hours(&one).into_inner();
    exact(
        &hp_h,
        335_564_942_212_021_599,
        125_000_000_000,
        "1 hp·h in J",
    );
    close(hp_h.eval_f64().unwrap(), 2_684_519.537696173, "1 hp·h in J");
}

/// 1 knot = 1 nmi/h and 1 mph = 1 mi/h exactly.
#[test]
fn speed_factors_compose_exactly() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let hour = Time::hours(&one).into_inner();
    let knot = Velocity::knots(&one).into_inner();
    let mph = Velocity::miles_per_hour(&one).into_inner();
    assert_eq!(
        (&knot * &hour)
            .eval()
            .equals(Length::nautical_miles(&one).inner()),
        Some(true)
    );
    assert_eq!(
        (&mph * &hour).eval().equals(Length::miles(&one).inner()),
        Some(true)
    );
    // 1 kn ≈ 1.150 779 mph (SP 811 B.8 gives kn → 1.852 km/h).
    close(
        knot.eval_f64().unwrap() / mph.eval_f64().unwrap(),
        1852.0 / 1609.344,
        "kn / mph",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Named-type forwarding methods generated by `define_quantity!`
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn named_type_simplify_full_and_rational_collapse_a_ratio() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x² − 1)/(x − 1) → x + 1
    let e = Length::from_ex(&(x.powi(2) - 1) / &(&x - 1));
    let want = &x + 1;
    assert_eq!(e.simplify_full().inner().equals(&want), Some(true));
    assert_eq!(e.simplify_rational().inner().equals(&want), Some(true));
    assert_eq!(e.cancel(&x).inner().equals(&want), Some(true));
}

#[test]
fn named_type_simplify_trig_uses_pythagoras() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = Dimensionless::from_ex(x.sin().powi(2) + x.cos().powi(2));
    assert_eq!(e.simplify_trig().inner().equals(&ctx.int(1)), Some(true));
}

#[test]
fn named_type_simplify_powers_merges_exponents() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = Area::from_ex(x.powi(2) * x.powi(3));
    assert_eq!(e.simplify_powers().inner().equals(&x.powi(5)), Some(true));
}

#[test]
fn named_type_expand_trig_and_trig_combine_round_trip() {
    let ctx = Context::new();
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let e = Dimensionless::from_ex((&a + &b).sin());
    let expanded = e.expand_trig();
    let want = &a.sin() * &b.cos() + &a.cos() * &b.sin();
    assert_eq!(expanded.inner().equals(&want), Some(true));
    // sin(a)cos(b) → ½[sin(a+b) + sin(a−b)]
    let prod = Dimensionless::from_ex(&a.sin() * &b.cos());
    let combined = prod.trig_combine();
    let want_c = &(&(&a + &b).sin() + &(&a - &b).sin()) / 2;
    assert_eq!(combined.inner().equals(&want_c), Some(true));
}

#[test]
fn named_type_expand_log_and_log_combine_round_trip() {
    let ctx = Context::new();
    let (a, b) = (
        ctx.symbol_with("a", &[Assumption::Positive]),
        ctx.symbol_with("b", &[Assumption::Positive]),
    );
    let e = Dimensionless::from_ex((&a * &b).ln());
    let want = &a.ln() + &b.ln();
    assert_eq!(e.expand_log().inner().equals(&want), Some(true));
    let back = Dimensionless::from_ex(want).log_combine();
    assert_eq!(back.inner().equals(&(&a * &b).ln()), Some(true));
}

#[test]
fn named_type_factor_collect_together_partial_fractions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // factor: x² − 1 = (x − 1)(x + 1)
    let f = Energy::from_ex(x.powi(2) - 1).factor(&x);
    assert_eq!(f.inner().equals(&(&(&x - 1) * &(&x + 1))), Some(true));
    // collect: x·a + x·b → x·(a + b) — checked by structure: one term.
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let c = Force::from_ex(&(&x * &a) + &(&x * &b)).collect(&x);
    assert_eq!(c.term_count(), 1);
    assert_eq!(c.inner().equals(&(&x * &(&a + &b))), Some(true));
    // together: 1/x + 1/(x+1) = (2x + 1)/(x(x+1))
    let t = Time::from_ex(&(1 / &x) + &(1 / &(&x + 1))).together();
    let want_t = &(2 * &x + 1) / &(&x * &(&x + 1));
    assert_eq!(t.inner().equals(&want_t), Some(true));
    // partial_fractions inverts together.
    let pf = Time::from_ex(want_t).partial_fractions(&x);
    assert_eq!(pf.term_count(), 2);
    assert_eq!(
        pf.inner().equals(&(&(1 / &x) + &(1 / &(&x + 1)))),
        Some(true)
    );
}

#[test]
fn named_type_rationalize_denom_clears_sqrt2() {
    let ctx = Context::new();
    let e = Length::from_ex(1 / &ctx.int(2).sqrt()).rationalize_denom();
    // 1/√2 = √2/2
    let want = &ctx.int(2).sqrt() / 2;
    assert_eq!(e.inner().equals(&want), Some(true));
    // The denominator is now rational: numerator/denominator split has
    // denominator 2.
    assert!(!format!("{}", e.inner()).contains("/sqrt"), "{}", e.inner());
}

#[test]
fn named_type_free_symbols_term_count_count_ops_is_zero_equals() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = Velocity::from_ex(&x.sin() + &(&y * 2));
    let mut fs: Vec<String> = e.free_symbols().iter().map(|s| s.to_string()).collect();
    fs.sort();
    assert_eq!(fs, ["x", "y"]);
    assert_eq!(e.term_count(), 2);
    // sin + Mul + Add
    assert_eq!(e.count_ops(), 3);
    assert!(!e.is_zero());
    assert!(Velocity::from_ex(&x - &x).is_zero());
    assert!(e.equals(&Velocity::from_ex(&(&y * 2) + &x.sin())));
    assert!(!e.equals(&Velocity::from_ex(x.sin())));
}

#[test]
fn named_type_eval_f64_with_and_eval_decimal() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = Power::from_ex(&(&x * &x) + &y);
    assert_eq!(e.eval_f64_with(&[(&x, 3), (&y, 4)]).unwrap(), 13.0);
    // 1/8 exactly.
    assert_eq!(
        Power::rational(&ctx, 1, 8).eval_decimal(10).unwrap(),
        "0.125"
    );
    // √2 to 20 digits: 1.4142135623730950488 (mpmath.sqrt(2) at 20 dp).
    assert_eq!(
        Power::from_ex(ctx.int(2).sqrt()).eval_decimal(20).unwrap(),
        "1.4142135623730950488"
    );
}

#[test]
fn named_type_checked_from_ex_rejects_wrong_dimension() {
    let ctx = Context::new();
    let (m, a) = (ctx.symbol("m"), ctx.symbol("a"));
    let dims = DimMap::new()
        .with("m", ConstDim::MASS)
        .with("a", ConstDim::ACCELERATION);
    let ok = Force::checked_from_ex(&m * &a, &dims).unwrap();
    assert!(ok.equals(&Force::from_ex(&m * &a)));
    let err = Energy::checked_from_ex(&m * &a, &dims).unwrap_err();
    assert!(
        err.contains("expected Energy [J]") && err.contains("inferred Force"),
        "{err}"
    );
}

#[test]
fn named_type_add_sub_neg_scalar_ownership_variants() {
    let ctx = Context::new();
    let a = Length::constant(&ctx, 3);
    let b = Length::constant(&ctx, 4);
    let x = ctx.symbol("x");
    exact(&(&a + &b), 7, 1, "&a + &b");
    exact(&(&a + b.clone()), 7, 1, "&a + b");
    exact(&(a.clone() + &b), 7, 1, "a + &b");
    exact(&(a.clone() + b.clone()), 7, 1, "a + b");
    exact(&(&a - &b), -1, 1, "&a - &b");
    exact(&(a.clone() - b.clone()), -1, 1, "a - b");
    exact(&(-a.clone()), -3, 1, "-a");
    exact(&(-&a), -3, 1, "-&a");
    exact(&(a.clone() * 5), 15, 1, "a * 5");
    exact(&(&a * 5), 15, 1, "&a * 5");
    exact(&(5 * a.clone()), 15, 1, "5 * a");
    exact(&(5 * &a), 15, 1, "5 * &a");
    exact(&(a.clone() / 2), 3, 2, "a / 2");
    assert_eq!((a.clone() * &x).inner().equals(&(3 * &x)), Some(true));
    assert_eq!((&a * &x).inner().equals(&(3 * &x)), Some(true));
    assert_eq!((&x * a.clone()).inner().equals(&(3 * &x)), Some(true));
}

#[test]
fn named_type_display_debug_dim_strings_and_zero() {
    let ctx = Context::new();
    let f = Force::rational(&ctx, 3, 2);
    assert_eq!(format!("{f}"), "3/2 [N]");
    assert_eq!(format!("{f:?}"), "Force(3/2)");
    assert_eq!(Force::dim_name_str(), "Force");
    assert_eq!(Force::dim_symbol_str(), "N");
    assert_eq!(Resistance::dim_symbol_str(), "Ω");
    assert!(Force::zero(&ctx).is_zero());
    assert_eq!(format!("{}", Torque::symbol(&ctx, "tau")), "tau [N·m]");
}

#[test]
fn named_type_cross_dimension_from_impls() {
    let ctx = Context::new();
    let f = Frequency::constant(&ctx, 50);
    let w: AngularVelocity = f.into();
    exact(&w, 50, 1, "Frequency → AngularVelocity");
    let f2: Frequency = w.into();
    exact(&f2, 50, 1, "AngularVelocity → Frequency");
    let f3 = Frequency::from_angular_velocity(AngularVelocity::constant(&ctx, 7));
    exact(&f3, 7, 1, "from_angular_velocity");
    let e = Energy::constant(&ctx, 9);
    let t: Torque = e.clone().into();
    exact(&t, 9, 1, "Energy → Torque");
    let e2: Energy = t.into();
    exact(&e2, 9, 1, "Torque → Energy");
    exact(&Torque::from_energy(e), 9, 1, "from_energy");
    exact(
        &Angle::from_dimensionless(Dimensionless::percent(&ctx.int(50))),
        1,
        2,
        "from_dimensionless",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Qty<D> — the generic wrapper's never-called methods
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn qty_map_applies_the_closure_and_keeps_the_dimension() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let q: Qty<LengthDim> = Qty::from_ex((&x + 1).powi(2));
    let m = q.map(|e| e.expand());
    let want = &(&x.powi(2) + &(2 * &x)) + 1;
    assert_eq!(m.inner().equals(&want), Some(true));
    assert_eq!(m.dim_name(), "Length");
    assert_eq!(m.dim_symbol(), "m");
}

#[test]
fn qty_simplify_expand_eval_forward_to_ex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let q: Qty<MassDim> = Qty::from_ex(&(&x + 1) * &(&x - 1));
    let want = &x.powi(2) - 1;
    assert_eq!(q.expand().inner().equals(&want), Some(true));
    assert_eq!(q.simplify().inner().equals(&want), Some(true));
    // eval folds sin(π) → 0.
    let s: Qty<MassDim> = Qty::from_ex(&ctx.pi().sin() + &ctx.int(2));
    exact(&s.eval(), 2, 1, "sin(π) + 2");
    exact(&s.subs(&ctx.pi(), &ctx.int(0)), 2, 1, "subs π→0");
}

#[test]
fn qty_simplify_trig_powers_rational_forward_to_ex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let t: Qty<DimensionlessDim> = Qty::from_ex(x.sin().powi(2) + x.cos().powi(2));
    exact(&t.simplify_trig(), 1, 1, "sin² + cos²");
    let p: Qty<VolumeDim> = Qty::from_ex(x.powi(2) * x.powi(3));
    assert_eq!(p.simplify_powers().inner().equals(&x.powi(5)), Some(true));
    let r: Qty<LengthDim> = Qty::from_ex(&(x.powi(2) - 1) / &(&x - 1));
    assert_eq!(r.simplify_rational().inner().equals(&(&x + 1)), Some(true));
    assert_eq!(r.cancel(&x).inner().equals(&(&x + 1)), Some(true));
}

#[test]
fn qty_expand_trig_log_and_combine_forward_to_ex() {
    let ctx = Context::new();
    let (a, b) = (
        ctx.symbol_with("a", &[Assumption::Positive]),
        ctx.symbol_with("b", &[Assumption::Positive]),
    );
    let s: Qty<DimensionlessDim> = Qty::from_ex((&a + &b).cos());
    let want = &(&a.cos() * &b.cos()) - &(&a.sin() * &b.sin());
    assert_eq!(s.expand_trig().inner().equals(&want), Some(true));
    let l: Qty<DimensionlessDim> = Qty::from_ex((&a / &b).ln());
    assert_eq!(
        l.expand_log().inner().equals(&(&a.ln() - &b.ln())),
        Some(true)
    );
    let lc: Qty<DimensionlessDim> = Qty::from_ex(&a.ln() + &b.ln());
    assert_eq!(lc.log_combine().inner().equals(&(&a * &b).ln()), Some(true));
    // cos(a)cos(b) → ½[cos(a−b) + cos(a+b)]
    let tc: Qty<DimensionlessDim> = Qty::from_ex(&a.cos() * &b.cos());
    let want_tc = &(&(&a - &b).cos() + &(&a + &b).cos()) / 2;
    assert_eq!(tc.trig_combine().inner().equals(&want_tc), Some(true));
}

#[test]
fn qty_factor_collect_together_partial_fractions_rationalize() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f: Qty<EnergyDim> = Qty::from_ex(&x.powi(2) + &(2 * &x) + 1);
    assert_eq!(f.factor(&x).inner().equals(&(&x + 1).powi(2)), Some(true));
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let c: Qty<ForceDim> = Qty::from_ex(&(&x * &a) + &(&x * &b));
    assert_eq!(c.collect(&x).term_count(), 1);
    let t: Qty<TimeDim> = Qty::from_ex(&(1 / &x) + &(1 / &(&x + 1)));
    let want_t = &(2 * &x + 1) / &(&x * &(&x + 1));
    assert_eq!(t.together().inner().equals(&want_t), Some(true));
    let pf: Qty<TimeDim> = Qty::from_ex(want_t);
    assert_eq!(pf.partial_fractions(&x).term_count(), 2);
    let r: Qty<LengthDim> = Qty::from_ex(1 / &ctx.int(3).sqrt());
    assert_eq!(
        r.rationalize_denom()
            .inner()
            .equals(&(&ctx.int(3).sqrt() / 3)),
        Some(true)
    );
}

#[test]
fn qty_diff_and_integrate_return_raw_ex() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    // x(t) = 3t² → v = 6t → a = 6.
    let x: Qty<LengthDim> = Qty::from_ex(3 * &t.powi(2));
    let v = x.diff(&t);
    assert_eq!(v.equals(&(6 * &t)), Some(true));
    let a: Qty<AccelerationDim> = Qty::from_ex(v.diff(&t));
    exact(&a, 6, 1, "a");
    // ∫ 6t dt = 3t² (no constant).
    let back: Qty<VelocityDim> = Qty::from_ex(6 * &t);
    assert_eq!(back.integrate(&t).equals(&(3 * &t.powi(2))), Some(true));
}

#[test]
fn qty_to_latex_free_symbols_contains_counts_is_zero() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let q: Qty<VoltageDim> = Qty::from_ex(&(&x / &y) + &x.sin());
    assert_eq!(q.to_latex(), q.inner().to_latex());
    assert!(q.to_latex().contains("\\frac{x}{y}"), "{}", q.to_latex());
    assert!(q.to_latex().contains("\\sin"), "{}", q.to_latex());
    let mut fs: Vec<String> = q.free_symbols().iter().map(|s| s.to_string()).collect();
    fs.sort();
    assert_eq!(fs, ["x", "y"]);
    assert!(q.contains(&x));
    assert!(!q.contains(&ctx.symbol("z")));
    assert_eq!(q.term_count(), 2);
    // Add, Mul(x, y^-1), Pow, Sin — the exact count of non-atom nodes.
    assert_eq!(q.count_ops(), 4);
    // x/y + sin(x) is not provably zero or nonzero; a literal 3 is nonzero.
    assert_ne!(q.is_zero(), Some(true));
    let three: Qty<VoltageDim> = Qty::from_ex(ctx.int(3));
    assert_eq!(three.is_zero(), Some(false));
    let z: Qty<VoltageDim> = Qty::from_ex(&x - &x);
    assert_eq!(z.is_zero(), Some(true));
}

#[test]
fn qty_eval_f64_with_and_eval_decimal() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let q: Qty<PowerDim> = Qty::from_ex(&(&x * &y) - 1);
    assert_eq!(q.eval_f64_with(&[(&x, 5), (&y, 7)]).unwrap(), 34.0);
    // y is left free, so the value is not a number.
    assert!(q.eval_f64_with(&[(&x, 5)]).is_err());
    let r: Qty<PowerDim> = Qty::from_ex(ctx.rational(1, 3));
    // 1/3 to 12 digits.
    assert_eq!(r.eval_decimal(12).unwrap(), "0.333333333333");
    assert!((r.eval_f64().unwrap() - 1.0 / 3.0).abs() < 1e-16);
}

#[test]
fn qty_dim_name_and_symbol_for_every_named_dim() {
    let ctx = Context::new();
    let e = ctx.int(1);
    macro_rules! check {
        ($($d:ty => $n:literal, $s:literal;)*) => {$(
            let q: Qty<$d> = Qty::from_ex(&e);
            assert_eq!(q.dim_name(), $n);
            assert_eq!(q.dim_symbol(), $s);
        )*};
    }
    check! {
        DimensionlessDim => "Dimensionless", "1";
        LengthDim => "Length", "m";
        MassDim => "Mass", "kg";
        TimeDim => "Time", "s";
        CurrentDim => "Current", "A";
        TemperatureDim => "Temperature", "K";
        AreaDim => "Area", "m²";
        VolumeDim => "Volume", "m³";
        VelocityDim => "Velocity", "m/s";
        AccelerationDim => "Acceleration", "m/s²";
        FrequencyDim => "Frequency", "Hz";
        AngularAccelerationDim => "AngularAcceleration", "rad/s²";
        ForceDim => "Force", "N";
        EnergyDim => "Energy", "J";
        PowerDim => "Power", "W";
        MomentumDim => "Momentum", "kg·m/s";
        AngularMomentumDim => "AngularMomentum", "kg·m²/s";
        MomentOfInertiaDim => "MomentOfInertia", "kg·m²";
        PressureDim => "Pressure", "Pa";
        StiffnessDim => "Stiffness", "N/m";
        DampingDim => "Damping", "N·s/m";
        VoltageDim => "Voltage", "V";
        ResistanceDim => "Resistance", "Ω";
        InductanceDim => "Inductance", "H";
        CapacitanceDim => "Capacitance", "F";
        ChargeDim => "Charge", "C";
        MagneticFluxDim => "MagneticFlux", "Wb";
    }
    // Aliases collapse onto the primary impl.
    let ang: Qty<AngleDim> = Qty::from_ex(&e);
    assert_eq!(ang.dim_name(), "Dimensionless");
    let tq: Qty<TorqueDim> = Qty::from_ex(&e);
    assert_eq!(tq.dim_symbol(), "J");
    let av: Qty<AngularVelocityDim> = Qty::from_ex(&e);
    assert_eq!(av.dim_symbol(), "Hz");
}

#[test]
fn qty_add_sub_neg_ownership_variants() {
    let ctx = Context::new();
    let a: Qty<LengthDim> = Qty::from_ex(ctx.int(3));
    let b: Qty<LengthDim> = Qty::from_ex(ctx.int(4));
    exact(&(a.clone() + b.clone()), 7, 1, "a + b");
    exact(&(a.clone() + &b), 7, 1, "a + &b");
    exact(&(&a + b.clone()), 7, 1, "&a + b");
    exact(&(&a + &b), 7, 1, "&a + &b");
    exact(&(a.clone() - b.clone()), -1, 1, "a - b");
    exact(&(a.clone() - &b), -1, 1, "a - &b");
    exact(&(&a - b.clone()), -1, 1, "&a - b");
    exact(&(&a - &b), -1, 1, "&a - &b");
    exact(&(-a.clone()), -3, 1, "-a");
    exact(&(-&a), -3, 1, "-&a");
}

#[test]
fn qty_mul_div_ownership_variants_compose_dimensions() {
    let ctx = Context::new();
    let f: Qty<ForceDim> = Qty::from_ex(ctx.int(6));
    let d: Qty<LengthDim> = Qty::from_ex(ctx.int(2));
    // Force × Length = Energy (dimension checked at compile time by the
    // annotation), value 12.
    let e1: Qty<EnergyDim> = f.clone() * d.clone();
    let e2: Qty<EnergyDim> = f.clone() * &d;
    let e3: Qty<EnergyDim> = &f * d.clone();
    let e4: Qty<EnergyDim> = &f * &d;
    for (i, e) in [e1, e2, e3, e4].iter().enumerate() {
        exact(e, 12, 1, &format!("F·d variant {i}"));
        assert_eq!(e.dim_name(), "Energy");
    }
    // Force / Length = Stiffness, value 3.
    let k1: Qty<StiffnessDim> = f.clone() / d.clone();
    let k2: Qty<StiffnessDim> = f.clone() / &d;
    let k3: Qty<StiffnessDim> = &f / d.clone();
    let k4: Qty<StiffnessDim> = &f / &d;
    for (i, k) in [k1, k2, k3, k4].iter().enumerate() {
        exact(k, 3, 1, &format!("F/d variant {i}"));
        assert_eq!(k.dim_symbol(), "N/m");
    }
}

#[test]
fn qty_scalar_i64_and_ex_ownership_variants() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let q: Qty<TimeDim> = Qty::from_ex(ctx.int(3));
    exact(&(q.clone() * 5), 15, 1, "q * 5");
    exact(&(&q * 5), 15, 1, "&q * 5");
    exact(&(5 * q.clone()), 15, 1, "5 * q");
    exact(&(5 * &q), 15, 1, "5 * &q");
    exact(&(q.clone() / 2), 3, 2, "q / 2");
    exact(&(&q / 2), 3, 2, "&q / 2");
    let want = 3 * &x;
    assert_eq!((q.clone() * &x).inner().equals(&want), Some(true));
    assert_eq!((&q * &x).inner().equals(&want), Some(true));
    assert_eq!((&x * q.clone()).inner().equals(&want), Some(true));
    assert_eq!((&x * &q).inner().equals(&want), Some(true));
    let want_d = 3 / &x;
    assert_eq!((q.clone() / &x).inner().equals(&want_d), Some(true));
    assert_eq!((&q / &x).inner().equals(&want_d), Some(true));
}

#[test]
fn assume_dimension_wraps_without_checking() {
    let ctx = Context::new();
    let raw = ctx.rational(9, 4);
    let f: Qty<ForceDim> = assume_dimension(raw.clone());
    exact(&f, 9, 4, "assume_dimension");
    assert_eq!(f.dim_name(), "Force");
    assert_eq!(format!("{f}"), "9/4");
    assert_eq!(format!("{f:?}"), "Qty(9/4)");
    // Round trip through the named type.
    let named: Force = f.into();
    assert_eq!(format!("{named}"), "9/4 [N]");
    let back: Qty<ForceDim> = named.as_qty();
    assert_eq!(back.into_inner(), raw);
}

// ═══════════════════════════════════════════════════════════════════════════
// ConstDim — name() / from_name() for every named dimension (dim.rs L292–419)
// ═══════════════════════════════════════════════════════════════════════════

const NAMED_DIMS: [(&str, ConstDim); 30] = [
    ("Dimensionless", ConstDim::DIMENSIONLESS),
    ("Angle", ConstDim::ANGLE),
    ("Length", ConstDim::LENGTH),
    ("Mass", ConstDim::MASS),
    ("Time", ConstDim::TIME),
    ("Current", ConstDim::CURRENT),
    ("Temperature", ConstDim::TEMPERATURE),
    ("Area", ConstDim::AREA),
    ("Volume", ConstDim::VOLUME),
    ("Velocity", ConstDim::VELOCITY),
    ("Acceleration", ConstDim::ACCELERATION),
    ("AngularVelocity", ConstDim::ANGULAR_VELOCITY),
    ("AngularAcceleration", ConstDim::ANGULAR_ACCELERATION),
    ("Frequency", ConstDim::FREQUENCY),
    ("Force", ConstDim::FORCE),
    ("Energy", ConstDim::ENERGY),
    ("Torque", ConstDim::TORQUE),
    ("Power", ConstDim::POWER),
    ("Momentum", ConstDim::MOMENTUM),
    ("AngularMomentum", ConstDim::ANGULAR_MOMENTUM),
    ("MomentOfInertia", ConstDim::MOMENT_OF_INERTIA),
    ("Pressure", ConstDim::PRESSURE),
    ("Stiffness", ConstDim::STIFFNESS),
    ("Damping", ConstDim::DAMPING),
    ("Voltage", ConstDim::VOLTAGE),
    ("Resistance", ConstDim::RESISTANCE),
    ("Inductance", ConstDim::INDUCTANCE),
    ("Capacitance", ConstDim::CAPACITANCE),
    ("Charge", ConstDim::CHARGE),
    ("MagneticFlux", ConstDim::MAGNETIC_FLUX),
];

#[test]
fn const_dim_from_name_returns_every_named_constant() {
    for (name, dim) in NAMED_DIMS {
        assert_eq!(ConstDim::from_name(name), Some(dim), "{name}");
    }
    assert_eq!(ConstDim::from_name("Entropy"), None);
    assert_eq!(ConstDim::from_name(""), None);
}

/// `name()` is the inverse of `from_name()` except on the three collision
/// groups, where the primary name wins (Angle→Dimensionless,
/// Torque→Energy, AngularVelocity→Frequency).
#[test]
fn const_dim_name_round_trips_except_aliases() {
    for (name, dim) in NAMED_DIMS {
        let want = match name {
            "Angle" => "Dimensionless",
            "Torque" => "Energy",
            "AngularVelocity" => "Frequency",
            n => n,
        };
        assert_eq!(dim.name(), want, "{name}");
        assert_eq!(format!("{dim}"), want, "Display of {name}");
    }
}

/// An exotic dimension has no name; `Display` falls back to exponents.
#[test]
fn const_dim_unknown_displays_raw_exponents() {
    // Boltzmann constant: J/K = L² M T⁻² Θ⁻¹
    let k_b = ConstDim::ENERGY.div(ConstDim::TEMPERATURE);
    assert_eq!(k_b, ConstDim::new(2, 1, -2, 0, -1, 0, 0));
    assert_eq!(k_b.name(), "Unknown");
    assert_eq!(format!("{k_b}"), "L^2 M^1 T^-2 I^0 Θ^-1 N^0 J^0");
    // pow scales every exponent.
    assert_eq!(ConstDim::LENGTH.pow(3), ConstDim::VOLUME);
    assert_eq!(
        ConstDim::VELOCITY.pow(-1),
        ConstDim::new(-1, 0, 1, 0, 0, 0, 0)
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// DimMap / inference — insert, check_dimensions and the untested node kinds
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dimmap_insert_is_the_in_place_form_of_with() {
    let mut dims = DimMap::new();
    assert!(dims.is_empty());
    dims.insert("m", ConstDim::MASS);
    dims.insert("a", ConstDim::ACCELERATION);
    assert_eq!(dims.len(), 2);
    assert_eq!(dims.get("m"), Some(&ConstDim::MASS));
    assert_eq!(dims.get("a"), Some(&ConstDim::ACCELERATION));
    assert_eq!(dims.get("x"), None);
    // Re-inserting overwrites.
    dims.insert("m", ConstDim::LENGTH);
    assert_eq!(dims.len(), 2);
    assert_eq!(dims.get("m"), Some(&ConstDim::LENGTH));
    let chained = DimMap::new()
        .with("m", ConstDim::LENGTH)
        .with("a", ConstDim::ACCELERATION);
    assert_eq!(chained.get("m"), dims.get("m"));
    assert_eq!(chained.get("a"), dims.get("a"));
}

#[test]
fn check_dimensions_accepts_consistent_and_rejects_mixed_sums() {
    let ctx = Context::new();
    let (m, a, x) = (ctx.symbol("m"), ctx.symbol("a"), ctx.symbol("x"));
    let dims = DimMap::new()
        .with_var(&m, ConstDim::MASS)
        .with_var(&a, ConstDim::ACCELERATION)
        .with_var(&x, ConstDim::LENGTH);
    assert_eq!(check_dimensions(&(&m * &a), &dims), Ok(()));
    assert_eq!(check_dimensions(&(&(&m * &a) * &x), &dims), Ok(()));
    let err = check_dimensions(&(&(&m * &a) + &x), &dims).unwrap_err();
    assert!(
        err.contains("Dimension mismatch in addition") && err.contains("Force"),
        "{err}"
    );
    let unknown = check_dimensions(&ctx.symbol("q"), &dims).unwrap_err();
    assert!(unknown.contains("Unknown variable 'q'"), "{unknown}");
}

#[test]
fn infer_dimension_handles_constants_neg_pow_and_transcendentals() {
    let ctx = Context::new();
    let (m, x, t) = (ctx.symbol("m"), ctx.symbol("x"), ctx.symbol("t"));
    let dims = symplex::units::constants::physical_constants_dimmap()
        .with("m", ConstDim::MASS)
        .with("x", ConstDim::LENGTH)
        .with("t", ConstDim::TIME);
    // A physical constant looked up by name: E = m c².
    let c = symplex::units::constants::speed_of_light(&ctx);
    let mc2 = &m * &c.inner().powi(2);
    assert_eq!(infer_dimension(&mc2, &dims), Ok(ConstDim::ENERGY));
    // π is dimensionless; −x keeps Length; x³ is Volume; x^(1/2) is rejected.
    assert_eq!(
        infer_dimension(&ctx.pi(), &dims),
        Ok(ConstDim::DIMENSIONLESS)
    );
    assert_eq!(infer_dimension(&(-&x), &dims), Ok(ConstDim::LENGTH));
    assert_eq!(infer_dimension(&x.powi(3), &dims), Ok(ConstDim::VOLUME));
    assert_eq!(
        infer_dimension(&x.powi(-2), &dims),
        Ok(ConstDim::new(-2, 0, 0, 0, 0, 0, 0))
    );
    let half = infer_dimension(&x.pow(&ctx.rational(1, 2)), &dims).unwrap_err();
    assert!(half.contains("Non-integer power"), "{half}");
    let bad_exp = infer_dimension(&ctx.int(2).pow(&x), &dims).unwrap_err();
    assert!(
        bad_exp.contains("Exponent must be dimensionless"),
        "{bad_exp}"
    );
    // sin of a dimensionless ratio is fine; sin of a length is not.
    assert_eq!(
        infer_dimension(&(&x / &x).sin(), &dims),
        Ok(ConstDim::DIMENSIONLESS)
    );
    let bad_sin = infer_dimension(&x.sin(), &dims).unwrap_err();
    assert!(bad_sin.contains("must be dimensionless"), "{bad_sin}");
    // An applied (opaque) function needs dimensionless arguments; its
    // formal derivative / integral divide / multiply by the variable's
    // dimension.  `s` is a dimensionless parameter, `x` a Length.
    let s = ctx.symbol("s");
    let dims = dims.with("s", ConstDim::DIMENSIONLESS);
    let f = ctx.apply("f", &[&s]);
    assert_eq!(infer_dimension(&f, &dims), Ok(ConstDim::DIMENSIONLESS));
    let bad_apply = infer_dimension(&ctx.apply("g", &[&t]), &dims).unwrap_err();
    assert!(
        bad_apply.contains("Applied function argument 0 must be dimensionless, got Time"),
        "{bad_apply}"
    );
    let xf = &x * &f;
    let d = xf.diff(&s);
    assert_eq!(d.expr_type(), ExprType::Mul, "{d}");
    assert!(format!("{d}").contains("Derivative(f(s), s)"), "{d}");
    assert_eq!(infer_dimension(&d, &dims), Ok(ConstDim::LENGTH));
    let i = xf.integrate(&s);
    assert!(format!("{i}").contains("Integral"), "{i}");
    assert_eq!(infer_dimension(&i, &dims), Ok(ConstDim::LENGTH));
    // x(t) = ½ g t² is a Length.
    let g = symplex::units::constants::standard_gravity(&ctx);
    let half_gt2 = &(&ctx.rational(1, 2) * g.inner()) * &t.powi(2);
    assert_eq!(assert_dimension(&half_gt2, &dims, ConstDim::LENGTH), Ok(()));
    let mism = assert_dimension(&half_gt2, &dims, ConstDim::TIME).unwrap_err();
    assert!(mism.contains("Expected dimension Time"), "{mism}");
}

/// `reduced_planck_constant` = h/(2π): the only never-called constant.
/// It displays as `hbar` and carries the action dimension.
#[test]
fn reduced_planck_constant_displays_as_hbar_with_action_dimension() {
    let ctx = Context::new();
    let hbar = symplex::units::constants::reduced_planck_constant(&ctx);
    assert_eq!(format!("{}", hbar.inner()), "hbar");
    assert_eq!(format!("{hbar}"), "hbar [kg·m²/s]");
    assert_eq!(hbar.as_qty().dim_name(), "AngularMomentum");
    // h/(2π) built by hand does evaluate (mpmath at 30 dp, 17 digits:
    // 1.0545718176461564e-34 J·s, CODATA 2018 / exact since 2019 SI).
    let h = symplex::units::constants::planck_constant(&ctx);
    let by_hand = h.inner() / &(2 * &ctx.pi());
    close(
        by_hand.eval_f64().unwrap(),
        1.0545718176461564e-34,
        "h/(2π)",
    );
}

/// `Context::physical_constant` documents that the constant "evaluates
/// numerically to `value`", and `reduced_planck_constant` that "both h and
/// π resolve to exact values".  Until 0.22 a constant whose value is a
/// composite expression (here `h/(2π)`; even `3/π`) failed in evalf with
/// `Unevaluable { reason: "sub-expression … not in cache" }` because the
/// constant is an atom to every tree walk; fixed in 0.22.1.
#[test]
fn reduced_planck_constant_evaluates_to_h_over_two_pi() {
    let ctx = Context::new();
    let hbar = symplex::units::constants::reduced_planck_constant(&ctx);
    close(hbar.eval_f64().unwrap(), 1.0545718176461564e-34, "ħ");
    let three_over_pi = ctx.physical_constant("q", &ctx.int(3) / &ctx.pi());
    close(
        three_over_pi.eval_f64().unwrap(),
        3.0 / PI,
        "3/π as a physical constant",
    );
}
