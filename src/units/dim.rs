//! Dimension vectors: phantom-typed compile-time SI dimension tracking.
//!
//! Each physical dimension is represented as a type-level 7-tuple of integers
//! (Length, Mass, Time, Current, Temperature, Amount, Luminosity) using
//! [`typenum`] type-level integers.

use core::marker::PhantomData;

use typenum::{N1, N2, N3, N4, P1, P2, P3, P4, Z0};

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
pub type DimensionlessDim = Dim<Z0, Z0, Z0, Z0, Z0, Z0, Z0>;
pub type AngleDim = Dim<Z0, Z0, Z0, Z0, Z0, Z0, Z0>; // same as Dimensionless

// --- Base SI ---
pub type LengthDim = Dim<P1, Z0, Z0, Z0, Z0, Z0, Z0>;
pub type MassDim = Dim<Z0, P1, Z0, Z0, Z0, Z0, Z0>;
pub type TimeDim = Dim<Z0, Z0, P1, Z0, Z0, Z0, Z0>;
pub type CurrentDim = Dim<Z0, Z0, Z0, P1, Z0, Z0, Z0>;
pub type TemperatureDim = Dim<Z0, Z0, Z0, Z0, P1, Z0, Z0>;

// --- Geometry ---
pub type AreaDim = Dim<P2, Z0, Z0, Z0, Z0, Z0, Z0>;
pub type VolumeDim = Dim<P3, Z0, Z0, Z0, Z0, Z0, Z0>;

// --- Kinematics ---
pub type VelocityDim = Dim<P1, Z0, N1, Z0, Z0, Z0, Z0>;
pub type AccelerationDim = Dim<P1, Z0, N2, Z0, Z0, Z0, Z0>;
pub type AngularVelocityDim = Dim<Z0, Z0, N1, Z0, Z0, Z0, Z0>; // same as FrequencyDim
pub type AngularAccelerationDim = Dim<Z0, Z0, N2, Z0, Z0, Z0, Z0>;
pub type FrequencyDim = Dim<Z0, Z0, N1, Z0, Z0, Z0, Z0>; // same as AngularVelocityDim

// --- Mechanics ---
pub type ForceDim = Dim<P1, P1, N2, Z0, Z0, Z0, Z0>;
pub type EnergyDim = Dim<P2, P1, N2, Z0, Z0, Z0, Z0>;
pub type TorqueDim = Dim<P2, P1, N2, Z0, Z0, Z0, Z0>; // same as EnergyDim
pub type PowerDim = Dim<P2, P1, N3, Z0, Z0, Z0, Z0>;
pub type MomentumDim = Dim<P1, P1, N1, Z0, Z0, Z0, Z0>;
pub type AngularMomentumDim = Dim<P2, P1, N1, Z0, Z0, Z0, Z0>;
pub type MomentOfInertiaDim = Dim<P2, P1, Z0, Z0, Z0, Z0, Z0>;
pub type PressureDim = Dim<N1, P1, N2, Z0, Z0, Z0, Z0>;
pub type StiffnessDim = Dim<Z0, P1, N2, Z0, Z0, Z0, Z0>;
pub type DampingDim = Dim<Z0, P1, N1, Z0, Z0, Z0, Z0>;

// --- Electromagnetism ---
pub type VoltageDim = Dim<P2, P1, N3, N1, Z0, Z0, Z0>;
pub type ResistanceDim = Dim<P2, P1, N3, N2, Z0, Z0, Z0>;
pub type InductanceDim = Dim<P2, P1, N2, N2, Z0, Z0, Z0>;
pub type CapacitanceDim = Dim<N2, N1, P4, P2, Z0, Z0, Z0>;
pub type ChargeDim = Dim<Z0, Z0, P1, P1, Z0, Z0, Z0>;
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
        Self { l, m, t, i, th, n, j }
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
}

// ---------------------------------------------------------------------------
// Named ConstDim constants
// ---------------------------------------------------------------------------

impl ConstDim {
    // Base / dimensionless
    pub const DIMENSIONLESS: Self = Self::new(0, 0, 0, 0, 0, 0, 0);
    pub const ANGLE: Self = Self::new(0, 0, 0, 0, 0, 0, 0);

    // Base SI
    pub const LENGTH: Self = Self::new(1, 0, 0, 0, 0, 0, 0);
    pub const MASS: Self = Self::new(0, 1, 0, 0, 0, 0, 0);
    pub const TIME: Self = Self::new(0, 0, 1, 0, 0, 0, 0);
    pub const CURRENT: Self = Self::new(0, 0, 0, 1, 0, 0, 0);
    pub const TEMPERATURE: Self = Self::new(0, 0, 0, 0, 1, 0, 0);

    // Geometry
    pub const AREA: Self = Self::new(2, 0, 0, 0, 0, 0, 0);
    pub const VOLUME: Self = Self::new(3, 0, 0, 0, 0, 0, 0);

    // Kinematics
    pub const VELOCITY: Self = Self::new(1, 0, -1, 0, 0, 0, 0);
    pub const ACCELERATION: Self = Self::new(1, 0, -2, 0, 0, 0, 0);
    pub const ANGULAR_VELOCITY: Self = Self::new(0, 0, -1, 0, 0, 0, 0);
    pub const ANGULAR_ACCELERATION: Self = Self::new(0, 0, -2, 0, 0, 0, 0);
    pub const FREQUENCY: Self = Self::new(0, 0, -1, 0, 0, 0, 0);

    // Mechanics
    pub const FORCE: Self = Self::new(1, 1, -2, 0, 0, 0, 0);
    pub const ENERGY: Self = Self::new(2, 1, -2, 0, 0, 0, 0);
    pub const TORQUE: Self = Self::new(2, 1, -2, 0, 0, 0, 0);
    pub const POWER: Self = Self::new(2, 1, -3, 0, 0, 0, 0);
    pub const MOMENTUM: Self = Self::new(1, 1, -1, 0, 0, 0, 0);
    pub const ANGULAR_MOMENTUM: Self = Self::new(2, 1, -1, 0, 0, 0, 0);
    pub const MOMENT_OF_INERTIA: Self = Self::new(2, 1, 0, 0, 0, 0, 0);
    pub const PRESSURE: Self = Self::new(-1, 1, -2, 0, 0, 0, 0);
    pub const STIFFNESS: Self = Self::new(0, 1, -2, 0, 0, 0, 0);
    pub const DAMPING: Self = Self::new(0, 1, -1, 0, 0, 0, 0);

    // Electromagnetism
    pub const VOLTAGE: Self = Self::new(2, 1, -3, -1, 0, 0, 0);
    pub const RESISTANCE: Self = Self::new(2, 1, -3, -2, 0, 0, 0);
    pub const INDUCTANCE: Self = Self::new(2, 1, -2, -2, 0, 0, 0);
    pub const CAPACITANCE: Self = Self::new(-2, -1, 4, 2, 0, 0, 0);
    pub const CHARGE: Self = Self::new(0, 0, 1, 1, 0, 0, 0);
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
