//! Dimension vectors: phantom-typed compile-time SI dimension tracking.
//!
//! Each physical dimension is represented as a type-level 7-tuple of integers
//! (Length, Mass, Time, Current, Temperature, Amount, Luminosity) using
//! [`typenum`] type-level integers.

use core::marker::PhantomData;

use typenum::{N1, N2, N3, P1, P2, P3, P4, Z0};

// ---------------------------------------------------------------------------
// Dim — phantom-typed dimension vector
// ---------------------------------------------------------------------------

/// A phantom type representing an SI dimension vector.
///
/// The seven type parameters correspond to the exponents of the seven SI base
/// dimensions:
///
/// | Parameter | Base dimension       |
/// |-----------|----------------------|
/// | `L`       | Length (m)           |
/// | `M`       | Mass (kg)            |
/// | `T`       | Time (s)             |
/// | `I`       | Electric current (A) |
/// | `Th`      | Temperature (K)      |
/// | `N`       | Amount of substance  |
/// | `J`       | Luminous intensity   |
#[derive(Clone, Copy)]
pub struct Dim<L, M, T, I, Th, N, J> {
    _phantom: PhantomData<(L, M, T, I, Th, N, J)>,
}

// ---------------------------------------------------------------------------
// Named dimension type aliases (30 total)
// ---------------------------------------------------------------------------

// --- Base / dimensionless ---
/// SI dimension for Dimensionless quantities: all exponents zero \[1\].
pub type DimensionlessDim = Dim<Z0, Z0, Z0, Z0, Z0, Z0, Z0>;
/// SI dimension for Angle: dimensionless [radian, rad]. Same as `DimensionlessDim`.
pub type AngleDim = Dim<Z0, Z0, Z0, Z0, Z0, Z0, Z0>; // same as Dimensionless

// --- Base SI ---
/// SI dimension for Length: L¹ [meter, m].
pub type LengthDim = Dim<P1, Z0, Z0, Z0, Z0, Z0, Z0>;
/// SI dimension for Mass: M¹ [kilogram, kg].
pub type MassDim = Dim<Z0, P1, Z0, Z0, Z0, Z0, Z0>;
/// SI dimension for Time: T¹ [second, s].
pub type TimeDim = Dim<Z0, Z0, P1, Z0, Z0, Z0, Z0>;
/// SI dimension for Electric Current: I¹ [ampere, A].
pub type CurrentDim = Dim<Z0, Z0, Z0, P1, Z0, Z0, Z0>;
/// SI dimension for Thermodynamic Temperature: Θ¹ [kelvin, K].
pub type TemperatureDim = Dim<Z0, Z0, Z0, Z0, P1, Z0, Z0>;

// --- Geometry ---
/// SI dimension for Area: L² [square meter, m²].
pub type AreaDim = Dim<P2, Z0, Z0, Z0, Z0, Z0, Z0>;
/// SI dimension for Volume: L³ [cubic meter, m³].
pub type VolumeDim = Dim<P3, Z0, Z0, Z0, Z0, Z0, Z0>;

// --- Kinematics ---
/// SI dimension for Velocity: L¹·T⁻¹ [meter per second, m/s].
pub type VelocityDim = Dim<P1, Z0, N1, Z0, Z0, Z0, Z0>;
/// SI dimension for Acceleration: L¹·T⁻² [meter per second squared, m/s²].
pub type AccelerationDim = Dim<P1, Z0, N2, Z0, Z0, Z0, Z0>;
/// SI dimension for Angular Velocity: T⁻¹ [radian per second, rad/s]. Same as `FrequencyDim`.
pub type AngularVelocityDim = Dim<Z0, Z0, N1, Z0, Z0, Z0, Z0>; // same as FrequencyDim
/// SI dimension for Angular Acceleration: T⁻² [radian per second squared, rad/s²].
pub type AngularAccelerationDim = Dim<Z0, Z0, N2, Z0, Z0, Z0, Z0>;
/// SI dimension for Frequency: T⁻¹ [hertz, Hz]. Same as `AngularVelocityDim`.
pub type FrequencyDim = Dim<Z0, Z0, N1, Z0, Z0, Z0, Z0>; // same as AngularVelocityDim

// --- Mechanics ---
/// SI dimension for Force: L¹·M¹·T⁻² [newton, N].
pub type ForceDim = Dim<P1, P1, N2, Z0, Z0, Z0, Z0>;
/// SI dimension for Energy: L²·M¹·T⁻² [joule, J].
pub type EnergyDim = Dim<P2, P1, N2, Z0, Z0, Z0, Z0>;
/// SI dimension for Torque: L²·M¹·T⁻² [newton-meter, N·m]. Same as `EnergyDim`.
pub type TorqueDim = Dim<P2, P1, N2, Z0, Z0, Z0, Z0>; // same as EnergyDim
/// SI dimension for Power: L²·M¹·T⁻³ [watt, W].
pub type PowerDim = Dim<P2, P1, N3, Z0, Z0, Z0, Z0>;
/// SI dimension for Momentum: L¹·M¹·T⁻¹ [kilogram meter per second, kg·m/s].
pub type MomentumDim = Dim<P1, P1, N1, Z0, Z0, Z0, Z0>;
/// SI dimension for Angular Momentum: L²·M¹·T⁻¹ [kilogram meter squared per second, kg·m²/s].
pub type AngularMomentumDim = Dim<P2, P1, N1, Z0, Z0, Z0, Z0>;
/// SI dimension for Moment of Inertia: L²·M¹ [kilogram meter squared, kg·m²].
pub type MomentOfInertiaDim = Dim<P2, P1, Z0, Z0, Z0, Z0, Z0>;
/// SI dimension for Pressure: L⁻¹·M¹·T⁻² [pascal, Pa].
pub type PressureDim = Dim<N1, P1, N2, Z0, Z0, Z0, Z0>;
/// SI dimension for Stiffness: M¹·T⁻² [newton per meter, N/m].
pub type StiffnessDim = Dim<Z0, P1, N2, Z0, Z0, Z0, Z0>;
/// SI dimension for Damping: M¹·T⁻¹ [newton-second per meter, N·s/m].
pub type DampingDim = Dim<Z0, P1, N1, Z0, Z0, Z0, Z0>;

// --- Electromagnetism ---
/// SI dimension for Voltage: L²·M¹·T⁻³·I⁻¹ [volt, V].
pub type VoltageDim = Dim<P2, P1, N3, N1, Z0, Z0, Z0>;
/// SI dimension for Resistance: L²·M¹·T⁻³·I⁻² [ohm, Ω].
pub type ResistanceDim = Dim<P2, P1, N3, N2, Z0, Z0, Z0>;
/// SI dimension for Inductance: L²·M¹·T⁻²·I⁻² [henry, H].
pub type InductanceDim = Dim<P2, P1, N2, N2, Z0, Z0, Z0>;
/// SI dimension for Capacitance: L⁻²·M⁻¹·T⁴·I² [farad, F].
pub type CapacitanceDim = Dim<N2, N1, P4, P2, Z0, Z0, Z0>;
/// SI dimension for Electric Charge: T¹·I¹ [coulomb, C].
pub type ChargeDim = Dim<Z0, Z0, P1, P1, Z0, Z0, Z0>;
/// SI dimension for Magnetic Flux: L²·M¹·T⁻²·I⁻¹ [weber, Wb].
pub type MagneticFluxDim = Dim<P2, P1, N2, N1, Z0, Z0, Z0>;

// ---------------------------------------------------------------------------
// DimName — human-readable names for dimension types
// ---------------------------------------------------------------------------

/// Provides human-readable names and symbols for dimension types.
///
/// This trait is intentionally **not** a bound on [`Dim`] or `Qty` — it is
/// only required on specific methods that format dimension information.
pub trait DimName {
    /// Full English name of the dimension (e.g. `"Force"`).
    fn dim_name() -> &'static str;
    /// SI symbol string (e.g. `"N"`, `"kg·m/s"`).
    fn dim_symbol() -> &'static str;
}

// ---------------------------------------------------------------------------
// DimName impls
//
// NOTE: Type aliases that resolve to the same concrete `Dim<…>` share a
// single impl.  Collision groups:
//   - DimensionlessDim ≡ AngleDim              → "Dimensionless" / "1"
//   - FrequencyDim ≡ AngularVelocityDim        → "Frequency" / "Hz"
//   - EnergyDim ≡ TorqueDim                    → "Energy" / "J"
//
// The *named newtypes* (Angle, Frequency, Torque, etc.) in the `qty` module
// are what provide distinct identity at the value level.
// ---------------------------------------------------------------------------

macro_rules! impl_dim_name {
    ($dim_ty:ty, $name:expr, $symbol:expr) => {
        impl DimName for $dim_ty {
            #[inline]
            fn dim_name() -> &'static str {
                $name
            }
            #[inline]
            fn dim_symbol() -> &'static str {
                $symbol
            }
        }
    };
}

// Dimensionless / Angle (all zeros)
impl_dim_name!(DimensionlessDim, "Dimensionless", "1");

// Base SI
impl_dim_name!(LengthDim, "Length", "m");
impl_dim_name!(MassDim, "Mass", "kg");
impl_dim_name!(TimeDim, "Time", "s");
impl_dim_name!(CurrentDim, "Current", "A");
impl_dim_name!(TemperatureDim, "Temperature", "K");

// Geometry
impl_dim_name!(AreaDim, "Area", "m²");
impl_dim_name!(VolumeDim, "Volume", "m³");

// Kinematics
impl_dim_name!(VelocityDim, "Velocity", "m/s");
impl_dim_name!(AccelerationDim, "Acceleration", "m/s²");
// FrequencyDim ≡ AngularVelocityDim
impl_dim_name!(FrequencyDim, "Frequency", "Hz");
impl_dim_name!(AngularAccelerationDim, "AngularAcceleration", "rad/s²");

// Mechanics
impl_dim_name!(ForceDim, "Force", "N");
// EnergyDim ≡ TorqueDim
impl_dim_name!(EnergyDim, "Energy", "J");
impl_dim_name!(PowerDim, "Power", "W");
impl_dim_name!(MomentumDim, "Momentum", "kg·m/s");
impl_dim_name!(AngularMomentumDim, "AngularMomentum", "kg·m²/s");
impl_dim_name!(MomentOfInertiaDim, "MomentOfInertia", "kg·m²");
impl_dim_name!(PressureDim, "Pressure", "Pa");
impl_dim_name!(StiffnessDim, "Stiffness", "N/m");
impl_dim_name!(DampingDim, "Damping", "N·s/m");

// Electromagnetism
impl_dim_name!(VoltageDim, "Voltage", "V");
impl_dim_name!(ResistanceDim, "Resistance", "Ω");
impl_dim_name!(InductanceDim, "Inductance", "H");
impl_dim_name!(CapacitanceDim, "Capacitance", "F");
impl_dim_name!(ChargeDim, "Charge", "C");
impl_dim_name!(MagneticFluxDim, "MagneticFlux", "Wb");

// ---------------------------------------------------------------------------
// ConstDim — const-evaluable dimension for compile-time assertions
// ---------------------------------------------------------------------------

/// A const-evaluable dimension vector for compile-time formula verification.
///
/// Unlike [`Dim`], which uses type-level integers, `ConstDim` stores exponents
/// as plain `i8` values so that dimension arithmetic can be performed in
/// `const fn` contexts (e.g. inside `const_assert_dim!`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ConstDim {
    /// Length exponent (m).
    pub l: i8,
    /// Mass exponent (kg).
    pub m: i8,
    /// Time exponent (s).
    pub t: i8,
    /// Electric current exponent (A).
    pub i: i8,
    /// Thermodynamic temperature exponent (K).
    pub th: i8,
    /// Amount of substance exponent (mol).
    pub n: i8,
    /// Luminous intensity exponent (cd).
    pub j: i8,
}

impl ConstDim {
    /// Create a new `ConstDim` with the given exponents.
    #[inline]
    pub const fn new(l: i8, m: i8, t: i8, i: i8, th: i8, n: i8, j: i8) -> Self {
        Self {
            l,
            m,
            t,
            i,
            th,
            n,
            j,
        }
    }

    /// Multiply two dimension vectors (add exponents).
    #[inline]
    pub const fn mul(self, rhs: Self) -> Self {
        Self {
            l: self.l + rhs.l,
            m: self.m + rhs.m,
            t: self.t + rhs.t,
            i: self.i + rhs.i,
            th: self.th + rhs.th,
            n: self.n + rhs.n,
            j: self.j + rhs.j,
        }
    }

    /// Divide two dimension vectors (subtract exponents).
    #[inline]
    pub const fn div(self, rhs: Self) -> Self {
        Self {
            l: self.l - rhs.l,
            m: self.m - rhs.m,
            t: self.t - rhs.t,
            i: self.i - rhs.i,
            th: self.th - rhs.th,
            n: self.n - rhs.n,
            j: self.j - rhs.j,
        }
    }

    /// Check equality of two dimension vectors (const-compatible).
    #[inline]
    pub const fn eq(self, rhs: Self) -> bool {
        self.l == rhs.l
            && self.m == rhs.m
            && self.t == rhs.t
            && self.i == rhs.i
            && self.th == rhs.th
            && self.n == rhs.n
            && self.j == rhs.j
    }

    /// Raise dimension to an integer power (multiply all exponents by `n`).
    #[inline]
    pub const fn pow(self, n: i8) -> Self {
        Self {
            l: self.l * n,
            m: self.m * n,
            t: self.t * n,
            i: self.i * n,
            th: self.th * n,
            n: self.n * n,
            j: self.j * n,
        }
    }

    /// Return a human-readable name for well-known dimensions, or a raw
    /// exponent string for exotic ones.
    pub fn name(&self) -> &'static str {
        // Check against all named constants. Order: dimensionless first,
        // then base SI, geometry, kinematics, mechanics, electromagnetism.
        if *self == Self::DIMENSIONLESS {
            return "Dimensionless";
        }
        if *self == Self::LENGTH {
            return "Length";
        }
        if *self == Self::MASS {
            return "Mass";
        }
        if *self == Self::TIME {
            return "Time";
        }
        if *self == Self::CURRENT {
            return "Current";
        }
        if *self == Self::TEMPERATURE {
            return "Temperature";
        }
        if *self == Self::AREA {
            return "Area";
        }
        if *self == Self::VOLUME {
            return "Volume";
        }
        if *self == Self::VELOCITY {
            return "Velocity";
        }
        if *self == Self::ACCELERATION {
            return "Acceleration";
        }
        if *self == Self::FREQUENCY {
            return "Frequency";
        }
        if *self == Self::ANGULAR_ACCELERATION {
            return "AngularAcceleration";
        }
        if *self == Self::FORCE {
            return "Force";
        }
        if *self == Self::ENERGY {
            return "Energy";
        }
        if *self == Self::POWER {
            return "Power";
        }
        if *self == Self::MOMENTUM {
            return "Momentum";
        }
        if *self == Self::ANGULAR_MOMENTUM {
            return "AngularMomentum";
        }
        if *self == Self::MOMENT_OF_INERTIA {
            return "MomentOfInertia";
        }
        if *self == Self::PRESSURE {
            return "Pressure";
        }
        if *self == Self::STIFFNESS {
            return "Stiffness";
        }
        if *self == Self::DAMPING {
            return "Damping";
        }
        if *self == Self::VOLTAGE {
            return "Voltage";
        }
        if *self == Self::RESISTANCE {
            return "Resistance";
        }
        if *self == Self::INDUCTANCE {
            return "Inductance";
        }
        if *self == Self::CAPACITANCE {
            return "Capacitance";
        }
        if *self == Self::CHARGE {
            return "Charge";
        }
        if *self == Self::MAGNETIC_FLUX {
            return "MagneticFlux";
        }
        // Exotic / unknown dimension — return a generic label.
        // (A const fn cannot format strings, so we return a fixed fallback.)
        "Unknown"
    }

    /// Look up a named dimension constant by its human-readable name.
    ///
    /// Returns `None` for unrecognized names. This is the inverse of
    /// [`name()`](Self::name) for all known dimensions.
    pub fn from_name(name: &str) -> Option<ConstDim> {
        match name {
            "Dimensionless" => Some(Self::DIMENSIONLESS),
            "Angle" => Some(Self::ANGLE),
            "Length" => Some(Self::LENGTH),
            "Mass" => Some(Self::MASS),
            "Time" => Some(Self::TIME),
            "Current" => Some(Self::CURRENT),
            "Temperature" => Some(Self::TEMPERATURE),
            "Area" => Some(Self::AREA),
            "Volume" => Some(Self::VOLUME),
            "Velocity" => Some(Self::VELOCITY),
            "Acceleration" => Some(Self::ACCELERATION),
            "AngularVelocity" => Some(Self::ANGULAR_VELOCITY),
            "AngularAcceleration" => Some(Self::ANGULAR_ACCELERATION),
            "Frequency" => Some(Self::FREQUENCY),
            "Force" => Some(Self::FORCE),
            "Energy" => Some(Self::ENERGY),
            "Torque" => Some(Self::TORQUE),
            "Power" => Some(Self::POWER),
            "Momentum" => Some(Self::MOMENTUM),
            "AngularMomentum" => Some(Self::ANGULAR_MOMENTUM),
            "MomentOfInertia" => Some(Self::MOMENT_OF_INERTIA),
            "Pressure" => Some(Self::PRESSURE),
            "Stiffness" => Some(Self::STIFFNESS),
            "Damping" => Some(Self::DAMPING),
            "Voltage" => Some(Self::VOLTAGE),
            "Resistance" => Some(Self::RESISTANCE),
            "Inductance" => Some(Self::INDUCTANCE),
            "Capacitance" => Some(Self::CAPACITANCE),
            "Charge" => Some(Self::CHARGE),
            "MagneticFlux" => Some(Self::MAGNETIC_FLUX),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Named ConstDim constants
// ---------------------------------------------------------------------------

impl ConstDim {
    // Base / dimensionless
    /// Dimensionless dimension: all exponents zero.
    pub const DIMENSIONLESS: Self = Self::new(0, 0, 0, 0, 0, 0, 0);
    /// Angle dimension: dimensionless (all exponents zero).
    pub const ANGLE: Self = Self::new(0, 0, 0, 0, 0, 0, 0);

    // Base SI
    /// Length dimension: L¹.
    pub const LENGTH: Self = Self::new(1, 0, 0, 0, 0, 0, 0);
    /// Mass dimension: M¹.
    pub const MASS: Self = Self::new(0, 1, 0, 0, 0, 0, 0);
    /// Time dimension: T¹.
    pub const TIME: Self = Self::new(0, 0, 1, 0, 0, 0, 0);
    /// Electric current dimension: I¹.
    pub const CURRENT: Self = Self::new(0, 0, 0, 1, 0, 0, 0);
    /// Temperature dimension: Θ¹.
    pub const TEMPERATURE: Self = Self::new(0, 0, 0, 0, 1, 0, 0);

    // Geometry
    /// Area dimension: L².
    pub const AREA: Self = Self::new(2, 0, 0, 0, 0, 0, 0);
    /// Volume dimension: L³.
    pub const VOLUME: Self = Self::new(3, 0, 0, 0, 0, 0, 0);

    // Kinematics
    /// Velocity dimension: L¹·T⁻¹.
    pub const VELOCITY: Self = Self::new(1, 0, -1, 0, 0, 0, 0);
    /// Acceleration dimension: L¹·T⁻².
    pub const ACCELERATION: Self = Self::new(1, 0, -2, 0, 0, 0, 0);
    /// Angular velocity dimension: T⁻¹.
    pub const ANGULAR_VELOCITY: Self = Self::new(0, 0, -1, 0, 0, 0, 0);
    /// Angular acceleration dimension: T⁻².
    pub const ANGULAR_ACCELERATION: Self = Self::new(0, 0, -2, 0, 0, 0, 0);
    /// Frequency dimension: T⁻¹.
    pub const FREQUENCY: Self = Self::new(0, 0, -1, 0, 0, 0, 0);

    // Mechanics
    /// Force dimension: L¹·M¹·T⁻².
    pub const FORCE: Self = Self::new(1, 1, -2, 0, 0, 0, 0);
    /// Energy dimension: L²·M¹·T⁻².
    pub const ENERGY: Self = Self::new(2, 1, -2, 0, 0, 0, 0);
    /// Torque dimension: L²·M¹·T⁻².
    pub const TORQUE: Self = Self::new(2, 1, -2, 0, 0, 0, 0);
    /// Power dimension: L²·M¹·T⁻³.
    pub const POWER: Self = Self::new(2, 1, -3, 0, 0, 0, 0);
    /// Momentum dimension: L¹·M¹·T⁻¹.
    pub const MOMENTUM: Self = Self::new(1, 1, -1, 0, 0, 0, 0);
    /// Angular momentum dimension: L²·M¹·T⁻¹.
    pub const ANGULAR_MOMENTUM: Self = Self::new(2, 1, -1, 0, 0, 0, 0);
    /// Moment of inertia dimension: L²·M¹.
    pub const MOMENT_OF_INERTIA: Self = Self::new(2, 1, 0, 0, 0, 0, 0);
    /// Pressure dimension: L⁻¹·M¹·T⁻².
    pub const PRESSURE: Self = Self::new(-1, 1, -2, 0, 0, 0, 0);
    /// Stiffness dimension: M¹·T⁻².
    pub const STIFFNESS: Self = Self::new(0, 1, -2, 0, 0, 0, 0);
    /// Damping dimension: M¹·T⁻¹.
    pub const DAMPING: Self = Self::new(0, 1, -1, 0, 0, 0, 0);

    // Electromagnetism
    /// Voltage dimension: L²·M¹·T⁻³·I⁻¹.
    pub const VOLTAGE: Self = Self::new(2, 1, -3, -1, 0, 0, 0);
    /// Resistance dimension: L²·M¹·T⁻³·I⁻².
    pub const RESISTANCE: Self = Self::new(2, 1, -3, -2, 0, 0, 0);
    /// Inductance dimension: L²·M¹·T⁻²·I⁻².
    pub const INDUCTANCE: Self = Self::new(2, 1, -2, -2, 0, 0, 0);
    /// Capacitance dimension: L⁻²·M⁻¹·T⁴·I².
    pub const CAPACITANCE: Self = Self::new(-2, -1, 4, 2, 0, 0, 0);
    /// Charge dimension: T¹·I¹.
    pub const CHARGE: Self = Self::new(0, 0, 1, 1, 0, 0, 0);
    /// Magnetic flux dimension: L²·M¹·T⁻²·I⁻¹.
    pub const MAGNETIC_FLUX: Self = Self::new(2, 1, -2, -1, 0, 0, 0);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_dim_mul_div_roundtrip() {
        let force = ConstDim::MASS.mul(ConstDim::ACCELERATION);
        assert!(force.eq(ConstDim::FORCE));

        let mass_back = ConstDim::FORCE.div(ConstDim::ACCELERATION);
        assert!(mass_back.eq(ConstDim::MASS));
    }

    #[test]
    fn const_dim_energy_from_force_times_length() {
        let energy = ConstDim::FORCE.mul(ConstDim::LENGTH);
        assert!(energy.eq(ConstDim::ENERGY));
    }

    #[test]
    fn const_dim_power_from_energy_div_time() {
        let power = ConstDim::ENERGY.div(ConstDim::TIME);
        assert!(power.eq(ConstDim::POWER));
    }

    #[test]
    fn const_dim_ohms_law() {
        // V = I * R  →  VoltageDim = CurrentDim * ResistanceDim
        let voltage = ConstDim::CURRENT.mul(ConstDim::RESISTANCE);
        assert!(voltage.eq(ConstDim::VOLTAGE));
    }

    #[test]
    fn const_dim_charge_from_current_times_time() {
        let charge = ConstDim::CURRENT.mul(ConstDim::TIME);
        assert!(charge.eq(ConstDim::CHARGE));
    }

    #[test]
    fn dim_name_force() {
        assert_eq!(ForceDim::dim_name(), "Force");
        assert_eq!(ForceDim::dim_symbol(), "N");
    }

    #[test]
    fn dim_name_voltage() {
        assert_eq!(VoltageDim::dim_name(), "Voltage");
        assert_eq!(VoltageDim::dim_symbol(), "V");
    }

    // Compile-time verification via const evaluation
    const _: () = {
        let f = ConstDim::MASS.mul(ConstDim::ACCELERATION);
        assert!(f.eq(ConstDim::FORCE));
    };
}
