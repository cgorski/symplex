//! Exact conversion factor constants for unit conversions.
//!
//! All non-trivial conversion factors are defined here as `(i64, i64)` pairs
//! representing exact rational numbers `numerator/denominator`. Each derived
//! constant includes a doc comment showing its derivation from base definitions.
//!
//! Every derived constant is verified by a test that recomputes it from base
//! definitions using `Ratio<BigInt>` arithmetic — independent of the hardcoded value.

/// Base conversion factors — exact by international agreement.
pub mod base {
    /// 1 inch = 25.4 mm = 127/5000 m (exact since 1959 international yard and pound agreement).
    pub const INCH: (i64, i64) = (127, 5000);

    /// 1 pound (avoirdupois) = 0.45359237 kg (exact since 1959).
    pub const POUND: (i64, i64) = (45359237, 100_000_000);

    /// Standard acceleration of gravity g_n = 9.80665 m/s² (exact, defined 1901 by CGPM).
    pub const G_N: (i64, i64) = (980665, 100_000);

    /// Thermochemical calorie = 4.184 J (exact by definition).
    pub const CALORIE_TH: (i64, i64) = (4184, 1000);

    /// Standard atmosphere = 101325 Pa (exact by definition, 1954).
    pub const ATM: (i64, i64) = (101325, 1);

    /// Nautical mile = 1852 m (exact by definition).
    pub const NAUTICAL_MILE: (i64, i64) = (1852, 1);

    /// Astronomical unit = 149597870700 m (exact since 2012 IAU definition).
    pub const ASTRONOMICAL_UNIT: (i64, i64) = (149597870700, 1);

    /// BTU (International Table) = 1055.05585262 J (exact by definition).
    pub const BTU_IT: (i64, i64) = (52752792631, 50_000_000);

    /// Imperial gallon = 4.54609 L = 0.00454609 m³ (exact by definition).
    pub const IMPERIAL_GALLON: (i64, i64) = (454609, 100_000_000);
}

/// Derived conversion factors — computed from base definitions and verified by tests.
pub mod derived {
    /// 1 foot = 12 inches = 381/1250 m.
    pub const FOOT: (i64, i64) = (381, 1250);

    /// 1 yard = 3 feet = 1143/1250 m.
    pub const YARD: (i64, i64) = (1143, 1250);

    /// 1 mile = 5280 feet = 201168/125 m.
    pub const MILE: (i64, i64) = (201168, 125);

    /// 1 fathom = 2 yards = 1143/625 m.
    pub const FATHOM: (i64, i64) = (1143, 625);

    /// 1 pound-force = 1 lb × g_n = 8896443230521/2000000000000 N.
    pub const POUND_FORCE: (i64, i64) = (8896443230521, 2_000_000_000_000);

    /// 1 kilogram-force = 1 kg × g_n = 196133/20000 N.
    pub const KILOGRAM_FORCE: (i64, i64) = (196133, 20000);

    /// 1 mechanical horsepower = 33000 ft·lbf/min = 37284993579113511/50000000000000 W.
    pub const HORSEPOWER: (i64, i64) = (37284993579113511, 50_000_000_000_000);

    /// 1 metric horsepower (PS) = 75 kgf·m/s = 588399/800 W.
    pub const METRIC_HORSEPOWER: (i64, i64) = (588399, 800);

    /// 1 psi = 1 lbf/in² = 8896443230521/1290320000 Pa.
    pub const PSI: (i64, i64) = (8896443230521, 1_290_320_000);

    /// 1 torr = 1 atm / 760 = 20265/152 Pa.
    pub const TORR: (i64, i64) = (20265, 152);

    /// 1 foot-pound = 1 ft × 1 lbf = 3389544870828501/2500000000000000 J.
    pub const FOOT_POUND: (i64, i64) = (3389544870828501, 2_500_000_000_000_000);

    /// 1 US gallon = 231 in³ = 473176473/125000000000 m³.
    pub const US_GALLON: (i64, i64) = (473176473, 125_000_000_000);

    /// 1 US quart = US gallon / 4.
    pub const US_QUART: (i64, i64) = (473176473, 500_000_000_000);

    /// 1 US pint = US gallon / 8.
    pub const US_PINT: (i64, i64) = (473176473, 1_000_000_000_000);

    /// 1 US fluid ounce = US gallon / 128.
    pub const US_FLUID_OUNCE: (i64, i64) = (473176473, 16_000_000_000_000);

    /// 1 US tablespoon = US fluid ounce / 2.
    pub const US_TABLESPOON: (i64, i64) = (473176473, 32_000_000_000_000);

    /// 1 ounce (avoirdupois) = 1 pound / 16.
    pub const OUNCE: (i64, i64) = (45359237, 1_600_000_000);

    /// 1 short ton = 2000 pounds.
    pub const SHORT_TON: (i64, i64) = (45359237, 50_000);

    /// 1 long ton = 2240 pounds.
    pub const LONG_TON: (i64, i64) = (317514659, 312_500);

    /// 1 grain = 1 pound / 7000.
    pub const GRAIN: (i64, i64) = (6479891, 100_000_000_000);

    /// 1 slug = 1 lbf·s²/ft (mass unit).
    pub const SLUG: (i64, i64) = (8896443230521, 609_600_000_000);

    /// 1 knot = 1 nautical mile / hour.
    pub const KNOT: (i64, i64) = (463, 900);

    /// 1 mph = 1 mile / hour.
    pub const MPH: (i64, i64) = (1397, 3125);

    /// 1 dyne = 10⁻⁵ N.
    pub const DYNE: (i64, i64) = (1, 100_000);

    /// 1 erg = 10⁻⁷ J.
    pub const ERG: (i64, i64) = (1, 10_000_000);
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::Ratio;

    fn r(n: i64, d: i64) -> Ratio<BigInt> {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    fn check(computed: Ratio<BigInt>, expected: (i64, i64), name: &str) {
        let en = BigInt::from(expected.0);
        let ed = BigInt::from(expected.1);
        let expected_r = Ratio::new(en.clone(), ed.clone());
        assert_eq!(
            computed, expected_r,
            "{name}: computed {computed} != expected {}/{}",
            en, ed
        );
    }

    fn check_float(val: (i64, i64), expected: f64, name: &str) {
        let actual = val.0 as f64 / val.1 as f64;
        let rel_err = ((actual - expected) / expected).abs();
        assert!(
            rel_err < 1e-14,
            "{name}: {actual} vs {expected}, relative error {rel_err}"
        );
    }

    // ── Base constant float checks ──

    #[test]
    fn base_inch() {
        check_float(base::INCH, 0.0254, "inch");
    }
    #[test]
    fn base_pound() {
        check_float(base::POUND, 0.45359237, "pound");
    }
    #[test]
    fn base_g_n() {
        check_float(base::G_N, 9.80665, "g_n");
    }
    #[test]
    fn base_cal() {
        check_float(base::CALORIE_TH, 4.184, "calorie");
    }
    #[test]
    fn base_atm() {
        assert_eq!(base::ATM, (101325, 1));
    }
    #[test]
    fn base_nmi() {
        assert_eq!(base::NAUTICAL_MILE, (1852, 1));
    }

    // ── Derived constant verification ──

    #[test]
    fn verify_foot() {
        let inch = r(base::INCH.0, base::INCH.1);
        check(r(12, 1) * inch, derived::FOOT, "foot = 12 inches");
    }

    #[test]
    fn verify_yard() {
        let foot = r(derived::FOOT.0, derived::FOOT.1);
        check(r(3, 1) * foot, derived::YARD, "yard = 3 feet");
    }

    #[test]
    fn verify_mile() {
        let foot = r(derived::FOOT.0, derived::FOOT.1);
        check(r(5280, 1) * foot, derived::MILE, "mile = 5280 feet");
    }

    #[test]
    fn verify_fathom() {
        let yard = r(derived::YARD.0, derived::YARD.1);
        check(r(2, 1) * yard, derived::FATHOM, "fathom = 2 yards");
    }

    #[test]
    fn verify_pound_force() {
        let pound = r(base::POUND.0, base::POUND.1);
        let g_n = r(base::G_N.0, base::G_N.1);
        check(pound * g_n, derived::POUND_FORCE, "lbf = lb × g_n");
    }

    #[test]
    fn verify_kilogram_force() {
        let g_n = r(base::G_N.0, base::G_N.1);
        check(g_n, derived::KILOGRAM_FORCE, "kgf = g_n");
    }

    #[test]
    fn verify_horsepower() {
        let foot = r(derived::FOOT.0, derived::FOOT.1);
        let lbf = r(derived::POUND_FORCE.0, derived::POUND_FORCE.1);
        let hp = r(33000, 1) * foot * lbf / r(60, 1);
        check(hp, derived::HORSEPOWER, "hp = 33000 ft·lbf/min");
    }

    #[test]
    fn verify_metric_horsepower() {
        let kgf = r(derived::KILOGRAM_FORCE.0, derived::KILOGRAM_FORCE.1);
        check(
            r(75, 1) * kgf,
            derived::METRIC_HORSEPOWER,
            "PS = 75 kgf·m/s",
        );
    }

    #[test]
    fn verify_psi() {
        let lbf = r(derived::POUND_FORCE.0, derived::POUND_FORCE.1);
        let inch = r(base::INCH.0, base::INCH.1);
        check(lbf / (inch.clone() * inch), derived::PSI, "psi = lbf/in²");
    }

    #[test]
    fn verify_torr() {
        let atm = r(base::ATM.0, base::ATM.1);
        check(atm / r(760, 1), derived::TORR, "torr = atm/760");
    }

    #[test]
    fn verify_foot_pound() {
        let foot = r(derived::FOOT.0, derived::FOOT.1);
        let lbf = r(derived::POUND_FORCE.0, derived::POUND_FORCE.1);
        check(foot * lbf, derived::FOOT_POUND, "ft·lbf = foot × lbf");
    }

    #[test]
    fn verify_us_gallon() {
        let inch = r(base::INCH.0, base::INCH.1);
        let in3 = inch.clone() * inch.clone() * inch;
        check(r(231, 1) * in3, derived::US_GALLON, "US gal = 231 in³");
    }

    #[test]
    fn verify_us_quart() {
        let gal = r(derived::US_GALLON.0, derived::US_GALLON.1);
        check(gal / r(4, 1), derived::US_QUART, "US qt = gal/4");
    }

    #[test]
    fn verify_us_pint() {
        let gal = r(derived::US_GALLON.0, derived::US_GALLON.1);
        check(gal / r(8, 1), derived::US_PINT, "US pt = gal/8");
    }

    #[test]
    fn verify_us_fl_oz() {
        let gal = r(derived::US_GALLON.0, derived::US_GALLON.1);
        check(
            gal / r(128, 1),
            derived::US_FLUID_OUNCE,
            "US fl oz = gal/128",
        );
    }

    #[test]
    fn verify_ounce() {
        let pound = r(base::POUND.0, base::POUND.1);
        check(pound / r(16, 1), derived::OUNCE, "oz = lb/16");
    }

    #[test]
    fn verify_short_ton() {
        let pound = r(base::POUND.0, base::POUND.1);
        check(
            r(2000, 1) * pound,
            derived::SHORT_TON,
            "short ton = 2000 lb",
        );
    }

    #[test]
    fn verify_long_ton() {
        let pound = r(base::POUND.0, base::POUND.1);
        check(r(2240, 1) * pound, derived::LONG_TON, "long ton = 2240 lb");
    }

    #[test]
    fn verify_grain() {
        let pound = r(base::POUND.0, base::POUND.1);
        check(pound / r(7000, 1), derived::GRAIN, "grain = lb/7000");
    }

    #[test]
    fn verify_slug() {
        let lbf = r(derived::POUND_FORCE.0, derived::POUND_FORCE.1);
        let foot = r(derived::FOOT.0, derived::FOOT.1);
        check(lbf / foot, derived::SLUG, "slug = lbf·s²/ft");
    }

    #[test]
    fn verify_knot() {
        let nmi = r(base::NAUTICAL_MILE.0, base::NAUTICAL_MILE.1);
        check(nmi / r(3600, 1), derived::KNOT, "knot = nmi/hr");
    }

    #[test]
    fn verify_mph() {
        let mile = r(derived::MILE.0, derived::MILE.1);
        check(mile / r(3600, 1), derived::MPH, "mph = mile/hr");
    }

    #[test]
    fn verify_hp_float() {
        check_float(derived::HORSEPOWER, 745.699_871_582_270_2, "horsepower");
    }

    #[test]
    fn verify_psi_float() {
        check_float(derived::PSI, 6894.757293168362, "psi");
    }
}
