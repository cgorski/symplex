//! Unit conversion constructors and getters for named quantity types.
//!
//! All quantities store values internally in SI base units.
//! Constructors convert from the specified unit to SI; getters convert back.

use super::conv_factors::{base, derived};
use super::si::*;
use crate::prelude::Ex;

// ===========================================================================
// Length [base unit: meter]
// ===========================================================================

impl Length {
    /// Create a Length from a value in meters (SI base unit).
    pub fn meters(val: &Ex) -> Self { Length(val.clone()) }

    /// Create a Length from a value in kilometers.
    pub fn kilometers(val: &Ex) -> Self { Length(val * 1000) }

    /// Create a Length from a value in centimeters.
    pub fn centimeters(val: &Ex) -> Self { Length(val * &crate::default_context().rational(1, 100)) }

    /// Create a Length from a value in millimeters.
    pub fn millimeters(val: &Ex) -> Self { Length(val * &crate::default_context().rational(1, 1000)) }

    /// Create a Length from a value in micrometers.
    pub fn micrometers(val: &Ex) -> Self { Length(val * &crate::default_context().rational(1, 1_000_000)) }

    /// Create a Length from a value in inches (1 in = 0.0254 m exactly).
    pub fn inches(val: &Ex) -> Self { Length(val * &crate::default_context().rational(base::INCH.0, base::INCH.1)) }

    /// Create a Length from a value in feet (1 ft = 0.3048 m exactly).
    pub fn feet(val: &Ex) -> Self { Length(val * &crate::default_context().rational(derived::FOOT.0, derived::FOOT.1)) }

    /// Create a Length from a value in yards (1 yd = 0.9144 m exactly).
    pub fn yards(val: &Ex) -> Self { Length(val * &crate::default_context().rational(derived::YARD.0, derived::YARD.1)) }

    /// Create a Length from a value in miles (1 mi = 1609.344 m exactly).
    pub fn miles(val: &Ex) -> Self { Length(val * &crate::default_context().rational(derived::MILE.0, derived::MILE.1)) }

    /// 1 nautical mile = 1852 m (exact).
    pub fn nautical_miles(val: &Ex) -> Self { Length(val * &crate::default_context().rational(base::NAUTICAL_MILE.0, base::NAUTICAL_MILE.1)) }

    /// 1 fathom = 2 yards (exact).
    pub fn fathoms(val: &Ex) -> Self { Length(val * &crate::default_context().rational(derived::FATHOM.0, derived::FATHOM.1)) }

    /// 1 mil = 0.001 inches (exact). Used in PCB design.
    pub fn mils(val: &Ex) -> Self { Length(val * &crate::default_context().rational(127, 5_000_000)) }
}

// ===========================================================================
// Mass [base unit: kilogram]
// ===========================================================================

impl Mass {
    /// Create a Mass from a value in kilograms (SI base unit).
    pub fn kilograms(val: &Ex) -> Self { Mass(val.clone()) }

    /// Create a Mass from a value in grams.
    pub fn grams(val: &Ex) -> Self { Mass(val * &crate::default_context().rational(1, 1000)) }

    /// Create a Mass from a value in milligrams.
    pub fn milligrams(val: &Ex) -> Self { Mass(val * &crate::default_context().rational(1, 1_000_000)) }

    /// Create a Mass from a value in metric tonnes (1 t = 1000 kg).
    pub fn tonnes(val: &Ex) -> Self { Mass(val * 1000) }

    /// Create a Mass from a value in pounds (1 lb = 0.45359237 kg exactly).
    pub fn pounds(val: &Ex) -> Self { Mass(val * &crate::default_context().rational(base::POUND.0, base::POUND.1)) }

    /// 1 ounce = 1/16 pound (exact).
    pub fn ounces(val: &Ex) -> Self { Mass(val * &crate::default_context().rational(derived::OUNCE.0, derived::OUNCE.1)) }

    /// 1 short ton = 2000 pounds (exact).
    pub fn short_tons(val: &Ex) -> Self { Mass(val * &crate::default_context().rational(derived::SHORT_TON.0, derived::SHORT_TON.1)) }

    /// 1 long ton = 2240 pounds (exact).
    pub fn long_tons(val: &Ex) -> Self { Mass(val * &crate::default_context().rational(derived::LONG_TON.0, derived::LONG_TON.1)) }

    /// 1 grain = 1/7000 pound (exact).
    pub fn grains(val: &Ex) -> Self { Mass(val * &crate::default_context().rational(derived::GRAIN.0, derived::GRAIN.1)) }

    /// 1 slug = 1 lbf·s²/ft (exact).
    pub fn slugs(val: &Ex) -> Self { Mass(val * &crate::default_context().rational(derived::SLUG.0, derived::SLUG.1)) }
}

// ===========================================================================
// Time [base unit: second]
// ===========================================================================

impl Time {
    /// Create a Time from a value in seconds (SI base unit).
    pub fn seconds(val: &Ex) -> Self { Time(val.clone()) }

    /// Create a Time from a value in milliseconds.
    pub fn milliseconds(val: &Ex) -> Self { Time(val * &crate::default_context().rational(1, 1000)) }

    /// Create a Time from a value in microseconds.
    pub fn microseconds(val: &Ex) -> Self { Time(val * &crate::default_context().rational(1, 1_000_000)) }

    /// Create a Time from a value in nanoseconds.
    pub fn nanoseconds(val: &Ex) -> Self { Time(val * &crate::default_context().rational(1, 1_000_000_000)) }

    /// Create a Time from a value in minutes (1 min = 60 s).
    pub fn minutes(val: &Ex) -> Self { Time(val * 60) }

    /// Create a Time from a value in hours (1 h = 3600 s).
    pub fn hours(val: &Ex) -> Self { Time(val * 3600) }

    /// Create a Time from a value in days (1 d = 86400 s).
    pub fn days(val: &Ex) -> Self { Time(val * 86400) }
}

// ===========================================================================
// Angle [base unit: radian]
// ===========================================================================

impl Angle {
    /// Create an Angle from a value in radians (SI base unit).
    pub fn radians(val: &Ex) -> Self { Angle(val.clone()) }

    /// Create an Angle from a value in degrees (1° = π/180 rad).
    pub fn degrees(val: &Ex) -> Self {
        Angle(val * &crate::default_context().pi() / 180)
    }

    /// Create an Angle from a value in revolutions (1 rev = 2π rad).
    pub fn revolutions(val: &Ex) -> Self {
        Angle(val * &(2 * &crate::default_context().pi()))
    }
}

// ===========================================================================
// Velocity [base unit: m/s]
// ===========================================================================

impl Velocity {
    /// Create a Velocity from a value in meters per second (SI base unit).
    pub fn meters_per_second(val: &Ex) -> Self { Velocity(val.clone()) }

    /// Create a Velocity from a value in kilometers per hour (1 km/h = 5/18 m/s).
    pub fn kilometers_per_hour(val: &Ex) -> Self {
        Velocity(val * &crate::default_context().rational(5, 18))
    }

    /// 1 mph = 1 mile / hour (exact).
    pub fn miles_per_hour(val: &Ex) -> Self { Velocity(val * &crate::default_context().rational(derived::MPH.0, derived::MPH.1)) }

    /// 1 knot = 1 nautical mile / hour (exact).
    pub fn knots(val: &Ex) -> Self { Velocity(val * &crate::default_context().rational(derived::KNOT.0, derived::KNOT.1)) }

    /// 1 ft/s (exact).
    pub fn feet_per_second(val: &Ex) -> Self { Velocity(val * &crate::default_context().rational(derived::FOOT.0, derived::FOOT.1)) }
}

// ===========================================================================
// Force [base unit: newton]
// ===========================================================================

impl Force {
    /// Create a Force from a value in newtons (SI base unit).
    pub fn newtons(val: &Ex) -> Self { Force(val.clone()) }

    /// Create a Force from a value in kilonewtons (1 kN = 1000 N).
    pub fn kilonewtons(val: &Ex) -> Self { Force(val * 1000) }

    /// 1 pound-force = lb × g_n (exact).
    pub fn pound_force(val: &Ex) -> Self { Force(val * &crate::default_context().rational(derived::POUND_FORCE.0, derived::POUND_FORCE.1)) }

    /// 1 kilogram-force = 1 kg × g_n (exact).
    pub fn kilogram_force(val: &Ex) -> Self { Force(val * &crate::default_context().rational(derived::KILOGRAM_FORCE.0, derived::KILOGRAM_FORCE.1)) }

    /// 1 dyne = 10⁻⁵ N (exact).
    pub fn dynes(val: &Ex) -> Self { Force(val * &crate::default_context().rational(derived::DYNE.0, derived::DYNE.1)) }
}

// ===========================================================================
// Energy [base unit: joule]
// ===========================================================================

impl Energy {
    /// Create an Energy from a value in joules (SI base unit).
    pub fn joules(val: &Ex) -> Self { Energy(val.clone()) }

    /// Create an Energy from a value in kilojoules (1 kJ = 1000 J).
    pub fn kilojoules(val: &Ex) -> Self { Energy(val * 1000) }

    /// Create an Energy from a value in kilowatt-hours (1 kWh = 3,600,000 J).
    pub fn kilowatt_hours(val: &Ex) -> Self { Energy(val * 3_600_000) }

    /// 1 thermochemical calorie = 4.184 J (exact).
    pub fn calories(val: &Ex) -> Self { Energy(val * &crate::default_context().rational(base::CALORIE_TH.0, base::CALORIE_TH.1)) }

    /// 1 kilocalorie = 4184 J (exact).
    pub fn kilocalories(val: &Ex) -> Self { Energy(val * 4184) }

    /// 1 BTU (International Table) = 1055.05585262 J (exact).
    pub fn btu(val: &Ex) -> Self { Energy(val * &crate::default_context().rational(base::BTU_IT.0, base::BTU_IT.1)) }

    /// 1 erg = 10⁻⁷ J (exact).
    pub fn ergs(val: &Ex) -> Self { Energy(val * &crate::default_context().rational(derived::ERG.0, derived::ERG.1)) }

    /// 1 foot-pound = 1 ft × 1 lbf (exact).
    pub fn foot_pounds(val: &Ex) -> Self { Energy(val * &crate::default_context().rational(derived::FOOT_POUND.0, derived::FOOT_POUND.1)) }
}

// ===========================================================================
// Power [base unit: watt]
// ===========================================================================

impl Power {
    /// Create a Power from a value in watts (SI base unit).
    pub fn watts(val: &Ex) -> Self { Power(val.clone()) }

    /// Create a Power from a value in kilowatts (1 kW = 1000 W).
    pub fn kilowatts(val: &Ex) -> Self { Power(val * 1000) }

    /// Create a Power from a value in megawatts (1 MW = 1,000,000 W).
    pub fn megawatts(val: &Ex) -> Self { Power(val * 1_000_000) }

    /// Create a Power from a value in mechanical horsepower.
    ///
    /// Exact definition: 1 hp = 33,000 ft·lbf/min = 37284993579113511/50000000000000 W.
    /// Derived from exact SI definitions: 1 ft = 381/1250 m, 1 lb = 45359237/100000000 kg,
    /// g_n = 980665/100000 m/s².
    pub fn horsepower(val: &Ex) -> Self {
        // Use the exact rational: 33000 × (381/1250) × (45359237/100000000) × (980665/100000) / 60
        // = 37284993579113511 / 50000000000000 W per hp (fits in i64)
        Power(val * &crate::default_context().rational(derived::HORSEPOWER.0, derived::HORSEPOWER.1))
    }

    /// 1 metric horsepower (PS) = 75 kgf·m/s (exact).
    pub fn metric_horsepower(val: &Ex) -> Self { Power(val * &crate::default_context().rational(derived::METRIC_HORSEPOWER.0, derived::METRIC_HORSEPOWER.1)) }
}

// ===========================================================================
// Voltage [base unit: volt]
// ===========================================================================

impl Voltage {
    /// Create a Voltage from a value in volts (SI base unit).
    pub fn volts(val: &Ex) -> Self { Voltage(val.clone()) }

    /// Create a Voltage from a value in millivolts (1 mV = 0.001 V).
    pub fn millivolts(val: &Ex) -> Self { Voltage(val * &crate::default_context().rational(1, 1000)) }

    /// Create a Voltage from a value in kilovolts (1 kV = 1000 V).
    pub fn kilovolts(val: &Ex) -> Self { Voltage(val * 1000) }
}

// ===========================================================================
// Current [base unit: ampere]
// ===========================================================================

impl Current {
    /// Create a Current from a value in amperes (SI base unit).
    pub fn amperes(val: &Ex) -> Self { Current(val.clone()) }

    /// Create a Current from a value in milliamperes (1 mA = 0.001 A).
    pub fn milliamperes(val: &Ex) -> Self { Current(val * &crate::default_context().rational(1, 1000)) }

    /// Create a Current from a value in microamperes (1 µA = 0.000001 A).
    pub fn microamperes(val: &Ex) -> Self { Current(val * &crate::default_context().rational(1, 1_000_000)) }
}

// ===========================================================================
// Resistance [base unit: ohm]
// ===========================================================================

impl Resistance {
    /// Create a Resistance from a value in ohms (SI base unit).
    pub fn ohms(val: &Ex) -> Self { Resistance(val.clone()) }

    /// Create a Resistance from a value in milliohms (1 mΩ = 0.001 Ω).
    pub fn milliohms(val: &Ex) -> Self { Resistance(val * &crate::default_context().rational(1, 1000)) }

    /// Create a Resistance from a value in kilohms (1 kΩ = 1000 Ω).
    pub fn kilohms(val: &Ex) -> Self { Resistance(val * 1000) }

    /// Create a Resistance from a value in megohms (1 MΩ = 1,000,000 Ω).
    pub fn megohms(val: &Ex) -> Self { Resistance(val * 1_000_000) }
}

// ===========================================================================
// Pressure [base unit: pascal]
// ===========================================================================

impl Pressure {
    /// Create a Pressure from a value in pascals (SI base unit).
    pub fn pascals(val: &Ex) -> Self { Pressure(val.clone()) }

    /// Create a Pressure from a value in kilopascals (1 kPa = 1000 Pa).
    pub fn kilopascals(val: &Ex) -> Self { Pressure(val * 1000) }

    /// Create a Pressure from a value in megapascals (1 MPa = 1,000,000 Pa).
    pub fn megapascals(val: &Ex) -> Self { Pressure(val * 1_000_000) }

    /// Create a Pressure from a value in bars (1 bar = 100,000 Pa).
    pub fn bars(val: &Ex) -> Self { Pressure(val * 100_000) }

    /// Create a Pressure from a value in standard atmospheres (1 atm = 101,325 Pa exactly).
    pub fn atmospheres(val: &Ex) -> Self { Pressure(val * 101_325) }

    /// 1 psi = 1 lbf/in² (exact).
    pub fn psi(val: &Ex) -> Self { Pressure(val * &crate::default_context().rational(derived::PSI.0, derived::PSI.1)) }

    /// 1 torr = 1 atm / 760 (exact).
    pub fn torr(val: &Ex) -> Self { Pressure(val * &crate::default_context().rational(derived::TORR.0, derived::TORR.1)) }
}

// ===========================================================================
// Frequency [base unit: hertz]
// ===========================================================================

impl Frequency {
    /// Create a Frequency from a value in hertz (SI base unit).
    pub fn hertz(val: &Ex) -> Self { Frequency(val.clone()) }

    /// Create a Frequency from a value in kilohertz (1 kHz = 1000 Hz).
    pub fn kilohertz(val: &Ex) -> Self { Frequency(val * 1000) }

    /// Create a Frequency from a value in megahertz (1 MHz = 1,000,000 Hz).
    pub fn megahertz(val: &Ex) -> Self { Frequency(val * 1_000_000) }

    /// Create a Frequency from a value in gigahertz (1 GHz = 1,000,000,000 Hz).
    pub fn gigahertz(val: &Ex) -> Self { Frequency(val * 1_000_000_000) }

    /// Create a Frequency from a value in revolutions per minute (1 rpm = 1/60 Hz).
    pub fn rpm(val: &Ex) -> Self { Frequency(val * &crate::default_context().rational(1, 60)) }

    /// 1 BPM = 1/60 Hz (beats per minute, used in medicine).
    pub fn bpm(val: &Ex) -> Self { Frequency(val * &crate::default_context().rational(1, 60)) }
}

// ===========================================================================
// Temperature [base unit: kelvin]
//
// NOTE: Celsius and Fahrenheit are *affine* (offset) conversions, not linear.
// Multiplying a temperature difference by a scalar is fine, but absolute
// temperature conversion requires an additive offset.
// ===========================================================================

impl Temperature {
    /// Create a Temperature from a value in kelvins (SI base unit).
    pub fn kelvins(val: &Ex) -> Self { Temperature(val.clone()) }

    /// Create a Temperature from a value in degrees Celsius (K = °C + 273.15).
    pub fn from_celsius(val: &Ex) -> Self {
        Temperature(val + &crate::default_context().rational(27315, 100))
    }

    /// Create a Temperature from a value in degrees Fahrenheit (K = (°F + 459.67) × 5/9).
    pub fn from_fahrenheit(val: &Ex) -> Self {
        Temperature(&(val + &crate::default_context().rational(45967, 100)) * &crate::default_context().rational(5, 9))
    }

    /// Convert from Rankine (absolute Fahrenheit scale). K = R × 5/9.
    pub fn from_rankine(val: &Ex) -> Self { Temperature(val * &crate::default_context().rational(5, 9)) }
}

// ===========================================================================
// Torque [base unit: N·m]
// ===========================================================================

impl Torque {
    /// Create a Torque from a value in newton-meters (SI base unit).
    pub fn newton_meters(val: &Ex) -> Self { Torque(val.clone()) }
}

// ===========================================================================
// Acceleration [base unit: m/s²]
// ===========================================================================

impl Acceleration {
    /// Create an Acceleration from a value in meters per second squared (SI base unit).
    pub fn meters_per_second_squared(val: &Ex) -> Self { Acceleration(val.clone()) }

    /// Create an Acceleration equal to standard gravity (9.80665 m/s² exactly).
    pub fn standard_gravity(val: &Ex) -> Self {
        Acceleration(val * &crate::default_context().rational(base::G_N.0, base::G_N.1))
    }

    /// 1 ft/s² (exact).
    pub fn feet_per_second_squared(val: &Ex) -> Self { Acceleration(val * &crate::default_context().rational(derived::FOOT.0, derived::FOOT.1)) }
}

// ===========================================================================
// Area [base unit: m²]
// ===========================================================================

impl Area {
    /// Create an Area from a value in square meters (SI base unit).
    pub fn square_meters(val: &Ex) -> Self { Area(val.clone()) }

    /// Create an Area from a value in square kilometers (1 km² = 1,000,000 m²).
    pub fn square_kilometers(val: &Ex) -> Self { Area(val * 1_000_000) }

    /// Create an Area from a value in square centimeters (1 cm² = 1/10000 m²).
    pub fn square_centimeters(val: &Ex) -> Self { Area(val * &crate::default_context().rational(1, 10_000)) }

    /// Create an Area from a value in hectares (1 ha = 10,000 m²).
    pub fn hectares(val: &Ex) -> Self { Area(val * 10_000) }
}

// ===========================================================================
// Volume [base unit: m³]
// ===========================================================================

impl Volume {
    /// Create a Volume from a value in cubic meters (SI base unit).
    pub fn cubic_meters(val: &Ex) -> Self { Volume(val.clone()) }

    /// Create a Volume from a value in liters (1 L = 0.001 m³).
    pub fn liters(val: &Ex) -> Self { Volume(val * &crate::default_context().rational(1, 1000)) }

    /// Create a Volume from a value in milliliters (1 mL = 1e-6 m³).
    pub fn milliliters(val: &Ex) -> Self { Volume(val * &crate::default_context().rational(1, 1_000_000)) }

    /// 1 US gallon = 231 in³ (exact).
    pub fn us_gallons(val: &Ex) -> Self { Volume(val * &crate::default_context().rational(derived::US_GALLON.0, derived::US_GALLON.1)) }

    /// 1 US quart = gallon/4 (exact).
    pub fn us_quarts(val: &Ex) -> Self { Volume(val * &crate::default_context().rational(derived::US_QUART.0, derived::US_QUART.1)) }

    /// 1 US pint = gallon/8 (exact).
    pub fn us_pints(val: &Ex) -> Self { Volume(val * &crate::default_context().rational(derived::US_PINT.0, derived::US_PINT.1)) }

    /// 1 US fluid ounce = gallon/128 (exact).
    pub fn us_fluid_ounces(val: &Ex) -> Self { Volume(val * &crate::default_context().rational(derived::US_FLUID_OUNCE.0, derived::US_FLUID_OUNCE.1)) }

    /// 1 imperial gallon = 4.54609 L (exact).
    pub fn imperial_gallons(val: &Ex) -> Self { Volume(val * &crate::default_context().rational(base::IMPERIAL_GALLON.0, base::IMPERIAL_GALLON.1)) }

    /// 1 US tablespoon = fl oz / 2 (exact).
    pub fn us_tablespoons(val: &Ex) -> Self { Volume(val * &crate::default_context().rational(derived::US_TABLESPOON.0, derived::US_TABLESPOON.1)) }
}

// ===========================================================================
// Momentum [base unit: kg·m/s]
// ===========================================================================

impl Momentum {
    /// Create a Momentum from a value in kilogram-meters per second (SI base unit).
    pub fn kilogram_meters_per_second(val: &Ex) -> Self { Momentum(val.clone()) }
}

// ===========================================================================
// AngularVelocity [base unit: rad/s]
// ===========================================================================

impl AngularVelocity {
    /// Create an AngularVelocity from a value in radians per second (SI base unit).
    pub fn radians_per_second(val: &Ex) -> Self { AngularVelocity(val.clone()) }

    /// Create an AngularVelocity from a value in RPM (1 rpm = 2π/60 rad/s).
    pub fn rpm(val: &Ex) -> Self {
        AngularVelocity(&(val * &crate::default_context().pi()) * &crate::default_context().rational(2, 60))
    }

    /// Create an AngularVelocity from a value in degrees per second.
    pub fn degrees_per_second(val: &Ex) -> Self {
        AngularVelocity(val * &crate::default_context().pi() / 180)
    }
}

// ===========================================================================
// Charge [base unit: coulomb]
// ===========================================================================

impl Charge {
    /// Create a Charge from a value in coulombs (SI base unit).
    pub fn coulombs(val: &Ex) -> Self { Charge(val.clone()) }

    /// Create a Charge from a value in milliampere-hours (1 mAh = 3.6 C).
    pub fn milliampere_hours(val: &Ex) -> Self { Charge(val * &crate::default_context().rational(36, 10)) }

    /// Create a Charge from a value in ampere-hours (1 Ah = 3600 C).
    pub fn ampere_hours(val: &Ex) -> Self { Charge(val * 3600) }
}

// ===========================================================================
// Capacitance [base unit: farad]
// ===========================================================================

impl Capacitance {
    /// Create a Capacitance from a value in farads (SI base unit).
    pub fn farads(val: &Ex) -> Self { Capacitance(val.clone()) }

    /// Create a Capacitance from a value in microfarads (1 µF = 1e-6 F).
    pub fn microfarads(val: &Ex) -> Self { Capacitance(val * &crate::default_context().rational(1, 1_000_000)) }

    /// Create a Capacitance from a value in nanofarads (1 nF = 1e-9 F).
    pub fn nanofarads(val: &Ex) -> Self {
        Capacitance(val * &crate::default_context().rational(1, 1_000_000_000))
    }

    /// Create a Capacitance from a value in picofarads (1 pF = 1e-12 F).
    pub fn picofarads(val: &Ex) -> Self {
        Capacitance(val * &crate::default_context().rational(1, 1_000_000_000_000))
    }
}

// ===========================================================================
// Inductance [base unit: henry]
// ===========================================================================

impl Inductance {
    /// Create an Inductance from a value in henrys (SI base unit).
    pub fn henrys(val: &Ex) -> Self { Inductance(val.clone()) }

    /// Create an Inductance from a value in millihenrys (1 mH = 0.001 H).
    pub fn millihenrys(val: &Ex) -> Self { Inductance(val * &crate::default_context().rational(1, 1000)) }

    /// Create an Inductance from a value in microhenrys (1 µH = 1e-6 H).
    pub fn microhenrys(val: &Ex) -> Self { Inductance(val * &crate::default_context().rational(1, 1_000_000)) }
}

// ===========================================================================
// MagneticFlux [base unit: weber]
// ===========================================================================

impl MagneticFlux {
    /// Create a MagneticFlux from a value in webers (SI base unit).
    pub fn webers(val: &Ex) -> Self { MagneticFlux(val.clone()) }
}

// ===========================================================================
// Stiffness [base unit: N/m]
// ===========================================================================

impl Stiffness {
    /// Create a Stiffness from a value in newtons per meter (SI base unit).
    pub fn newtons_per_meter(val: &Ex) -> Self { Stiffness(val.clone()) }

    /// Create a Stiffness from a value in kilonewtons per meter (1 kN/m = 1000 N/m).
    pub fn kilonewtons_per_meter(val: &Ex) -> Self { Stiffness(val * 1000) }
}

// ===========================================================================
// Damping [base unit: N·s/m]
// ===========================================================================

impl Damping {
    /// Create a Damping from a value in newton-seconds per meter (SI base unit).
    pub fn newton_seconds_per_meter(val: &Ex) -> Self { Damping(val.clone()) }
}

// ===========================================================================
// Dimensionless
// ===========================================================================

impl Dimensionless {
    /// Create a Dimensionless quantity from a value in percent (1% = 0.01).
    pub fn percent(val: &Ex) -> Self { Dimensionless(val * &crate::default_context().rational(1, 100)) }

    /// Create a Dimensionless quantity from a value in parts per thousand (‰).
    pub fn per_mille(val: &Ex) -> Self { Dimensionless(val * &crate::default_context().rational(1, 1000)) }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_meters_identity() {
        let v = crate::default_context().int(5);
        let l = Length::meters(&v);
        assert_eq!(l.into_inner().eval().to_string(), "5");
    }

    #[test]
    fn length_kilometers_scales() {
        let v = crate::default_context().int(3);
        let l = Length::kilometers(&v);
        assert_eq!(l.into_inner().eval().to_string(), "3000");
    }

    #[test]
    fn length_centimeters_scales() {
        let v = crate::default_context().int(200);
        let l = Length::centimeters(&v);
        assert_eq!(l.into_inner().eval().to_string(), "2");
    }

    #[test]
    fn length_inches_exact() {
        let v = crate::default_context().int(1);
        let l = Length::inches(&v);
        assert_eq!(l.into_inner().eval().to_string(), "127/5000");
    }

    #[test]
    fn mass_pounds_exact() {
        let v = crate::default_context().int(1);
        let m = Mass::pounds(&v);
        assert_eq!(m.into_inner().eval().to_string(), "45359237/100000000");
    }

    #[test]
    fn time_hours_scales() {
        let v = crate::default_context().int(2);
        let t = Time::hours(&v);
        assert_eq!(t.into_inner().eval().to_string(), "7200");
    }

    #[test]
    fn temperature_from_celsius() {
        // 0°C = 273.15 K = 27315/100 = 5463/20 (simplified)
        let v = crate::default_context().int(0);
        let t = Temperature::from_celsius(&v);
        let result = t.into_inner().eval().to_string();
        // Accept either simplified or unsimplified form
        assert!(
            result == "5463/20" || result == "27315/100",
            "Expected 5463/20 or 27315/100, got {result}"
        );
    }

    #[test]
    fn temperature_from_fahrenheit_boiling() {
        // 212°F should be 373.15 K = 37315/100 = 7463/20 (simplified)
        let v = crate::default_context().int(212);
        let t = Temperature::from_fahrenheit(&v);
        let result = t.into_inner().eval().to_string();
        assert!(
            result == "7463/20" || result == "37315/100",
            "Expected 7463/20 or 37315/100, got {result}"
        );
    }

    #[test]
    fn pressure_atmospheres_exact() {
        let v = crate::default_context().int(1);
        let p = Pressure::atmospheres(&v);
        assert_eq!(p.into_inner().eval().to_string(), "101325");
    }

    #[test]
    fn velocity_kmh_to_ms() {
        let v = crate::default_context().int(90);
        let vel = Velocity::kilometers_per_hour(&v);
        assert_eq!(vel.into_inner().eval().to_string(), "25");
    }

    #[test]
    fn energy_kwh_to_joules() {
        let v = crate::default_context().int(1);
        let e = Energy::kilowatt_hours(&v);
        assert_eq!(e.into_inner().eval().to_string(), "3600000");
    }

    #[test]
    fn frequency_rpm_to_hertz() {
        let v = crate::default_context().int(120);
        let f = Frequency::rpm(&v);
        assert_eq!(f.into_inner().eval().to_string(), "2");
    }

    #[test]
    fn symbolic_length_preserves_variable() {
        let x = crate::default_context().symbol("x");
        let l = Length::kilometers(&x);
        let inner = l.into_inner();
        assert!(inner.to_string().contains("x"));
    }

    #[test]
    fn volume_liters_scales() {
        let v = crate::default_context().int(1000);
        let vol = Volume::liters(&v);
        assert_eq!(vol.into_inner().eval().to_string(), "1");
    }

    #[test]
    fn dimensionless_percent() {
        let v = crate::default_context().int(50);
        let d = Dimensionless::percent(&v);
        assert_eq!(d.into_inner().eval().to_string(), "1/2");
    }

    #[test]
    fn charge_ampere_hours() {
        let v = crate::default_context().int(1);
        let c = Charge::ampere_hours(&v);
        assert_eq!(c.into_inner().eval().to_string(), "3600");
    }

    #[test]
    fn area_hectares() {
        let v = crate::default_context().int(2);
        let a = Area::hectares(&v);
        assert_eq!(a.into_inner().eval().to_string(), "20000");
    }
}
