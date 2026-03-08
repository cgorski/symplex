//! Named newtype wrappers for 30 SI physical quantities.
//!
//! Each newtype wraps an [`Ex`] and provides compile-time dimensional safety
//! with clear error messages (e.g. "expected `Force`, found `Mass`").
//!
//! Dimension-colliding pairs share the same underlying `Dim<…>` type alias:
//!
//! | Primary          | Alias              | Dim vector            |
//! |------------------|--------------------|-----------------------|
//! | `Dimensionless`  | `Angle`            | `<Z0,Z0,Z0,Z0,…>`   |
//! | `Energy`         | `Torque`           | `<P2,P1,N2,Z0,…>`   |
//! | `AngularVelocity`| `Frequency`        | `<Z0,Z0,N1,Z0,…>`   |
//!
//! Only the primary type in each group gets `From<Qty<D>>`. The alias type
//! provides a named conversion method instead to avoid conflicting impls.

use std::fmt;
use std::ops;

use crate::prelude::Ex;
use super::dim::*;
use super::qty::Qty;

// ===========================================================================
// define_quantity! — generates a named newtype plus all trait impls
// ===========================================================================

macro_rules! define_quantity {
    // ------------------------------------------------------------------
    // Default arm: includes From<Qty<$dim>> for $name
    // ------------------------------------------------------------------
    (
        $(#[$meta:meta])*
        $name:ident, $dim:ty, $dim_name_str:literal, $dim_sym:literal
    ) => {
        define_quantity!(@common $(#[$meta])* $name, $dim, $dim_name_str, $dim_sym);

        impl From<Qty<$dim>> for $name {
            fn from(q: Qty<$dim>) -> $name { $name(q.inner) }
        }

        impl $crate::units::qty::FromDimExpr<$dim> for $name {
            fn from_dim_expr(qty: $crate::units::qty::Qty<$dim>) -> Self {
                $name(qty.inner)
            }
        }
    };

    // ------------------------------------------------------------------
    // @no_qty_from arm: omits From<Qty<$dim>> for $name (collision avoidance)
    // ------------------------------------------------------------------
    (
        @no_qty_from
        $(#[$meta:meta])*
        $name:ident, $dim:ty, $dim_name_str:literal, $dim_sym:literal
    ) => {
        define_quantity!(@common $(#[$meta])* $name, $dim, $dim_name_str, $dim_sym);
    };

    // ------------------------------------------------------------------
    // @common — shared implementation for both arms
    // ------------------------------------------------------------------
    (
        @common
        $(#[$meta:meta])*
        $name:ident, $dim:ty, $dim_name_str:literal, $dim_sym:literal
    ) => {
        $(#[$meta])*
        #[derive(Clone)]
        pub struct $name(pub(crate) Ex);

        impl $name {
            /// Create from a symbolic expression (or anything that implements `IntoEx`).
            pub fn from_ex(ex: impl $crate::units::qty::IntoEx) -> Self { $name(ex.into_ex()) }

            /// Create from a symbolic expression with runtime dimension validation.
            ///
            /// Checks that the expression's inferred dimension matches this type's
            /// dimension using the provided [`DimMap`]. Returns an error if the
            /// dimensions don't match.
            ///
            /// Use this instead of `from_ex` when you want runtime validation
            /// of dimensional correctness (e.g., in tests or debug builds).
            pub fn checked_from_ex(
                ex: impl $crate::units::qty::IntoEx,
                dims: &$crate::units::inference::DimMap,
            ) -> Result<Self, String> {
                let ex = ex.into_ex();
                let inferred = $crate::units::inference::infer_dimension(&ex, dims)?;
                if let Some(expected) = $crate::units::dim::ConstDim::from_name($dim_name_str) {
                    if !inferred.eq(expected) {
                        return Err(format!(
                            "Dimension mismatch: expected {} [{}], inferred {}",
                            $dim_name_str, $dim_sym, inferred
                        ));
                    }
                }
                Ok($name(ex))
            }

            /// Create a named symbolic variable with this dimension.
            pub fn symbol(name: &str) -> Self {
                $name(crate::default_context().symbol(name))
            }

            /// Create from an integer constant.
            pub fn constant(val: i64) -> Self {
                $name(crate::default_context().int(val))
            }

            /// Create from a rational constant.
            pub fn rational(p: i64, q: i64) -> Self {
                $name(crate::default_context().rational(p, q))
            }

            /// Zero value.
            pub fn zero() -> Self { $name(crate::default_context().int(0)) }

            /// Escape hatch: drop dimension, return raw Ex.
            pub fn into_inner(self) -> Ex { self.0 }

            /// Borrow the inner Ex.
            pub fn inner(&self) -> &Ex { &self.0 }

            /// Convert to generic Qty<D>.
            pub fn as_qty(self) -> Qty<$dim> { Qty::from_ex(self.0) }

            /// Runtime dimension name.
            pub fn dim_name_str() -> &'static str { $dim_name_str }

            /// Runtime dimension symbol.
            pub fn dim_symbol_str() -> &'static str { $dim_sym }

            /// Apply simplify, preserving dimension.
            pub fn simplify(&self) -> Self { $name(self.0.simplify()) }

            /// Apply expand, preserving dimension.
            pub fn expand(&self) -> Self { $name(self.0.expand()) }

            /// Apply eval, preserving dimension.
            pub fn eval(&self) -> Self { $name(self.0.eval()) }

            /// Substitute a variable.
            pub fn subs(&self, var: &impl AsRef<$crate::prelude::Ex>, val: &impl AsRef<$crate::prelude::Ex>) -> Self { $name(self.0.subs(var.as_ref(), val.as_ref())) }

            // ── Dimension-preserving manipulation ──────────────────────

            /// Full multi-pass simplification, preserving dimension.
            pub fn simplify_full(&self) -> Self { $name(self.0.full_simplify()) }

            /// Trigonometric simplification, preserving dimension.
            pub fn simplify_trig(&self) -> Self { $name(self.0.simplify_trig()) }

            /// Power/exponent simplification, preserving dimension.
            pub fn simplify_powers(&self) -> Self { $name(self.0.simplify_powers()) }

            /// Rational simplification, preserving dimension.
            pub fn simplify_rational(&self) -> Self { $name(self.0.simplify_rational()) }

            /// Expand trigonometric identities, preserving dimension.
            pub fn expand_trig(&self) -> Self { $name(self.0.expand_trig()) }

            /// Expand logarithmic identities, preserving dimension.
            pub fn expand_log(&self) -> Self { $name(self.0.expand_log()) }

            /// Combine logarithmic terms, preserving dimension.
            pub fn log_combine(&self) -> Self { $name(self.0.log_combine()) }

            /// Combine trigonometric terms, preserving dimension.
            pub fn trig_combine(&self) -> Self { $name(self.0.trig_combine()) }

            /// Factor with respect to a variable, preserving dimension.
            pub fn factor(&self, var: &impl AsRef<$crate::prelude::Ex>) -> Self { $name(self.0.factor(var.as_ref())) }

            /// Collect terms with respect to a variable, preserving dimension.
            pub fn collect(&self, var: &impl AsRef<$crate::prelude::Ex>) -> Self { $name(self.0.collect(var.as_ref())) }

            /// Cancel common factors, preserving dimension.
            pub fn cancel(&self, var: &Ex) -> Self { $name(self.0.cancel(var)) }

            /// Combine fractions over a common denominator, preserving dimension.
            pub fn together(&self) -> Self { $name(self.0.together()) }

            /// Partial-fraction decomposition with respect to a variable, preserving dimension.
            pub fn partial_fractions(&self, var: &impl AsRef<$crate::prelude::Ex>) -> Self { $name(self.0.partial_fractions(var.as_ref())) }

            /// Rationalize the denominator, preserving dimension.
            pub fn rationalize_denom(&self) -> Self { $name(self.0.rationalize_denom()) }

            // ── Calculus — returns raw Ex ──────────────────────────────

            /// Differentiate with respect to a variable. Returns raw Ex.
            /// Wrap result in the correct output type: `Acceleration::from_ex(v.diff(&t))`
            pub fn diff(&self, var: &impl AsRef<$crate::prelude::Ex>) -> Ex { self.0.diff(var.as_ref()) }

            /// Integrate with respect to a variable. Returns raw Ex.
            pub fn integrate(&self, var: &impl AsRef<$crate::prelude::Ex>) -> Ex { self.0.integrate(var.as_ref()) }

            // ── Queries ───────────────────────────────────────────────

            /// Render as LaTeX string.
            pub fn to_latex(&self) -> String { self.0.to_latex() }

            /// Free symbols in the expression.
            pub fn free_symbols(&self) -> Vec<Ex> { self.0.free_symbols() }

            /// Whether the expression contains a given sub-expression.
            pub fn contains(&self, other: &impl AsRef<$crate::prelude::Ex>) -> bool { self.0.contains(other.as_ref()) }

            /// Number of top-level additive terms.
            pub fn term_count(&self) -> usize { self.0.term_count() }

            /// Count of internal operations.
            pub fn count_ops(&self) -> usize { self.0.count_ops() }

            /// Check whether this quantity is provably zero.
            pub fn is_zero(&self) -> bool { self.0.is_zero().unwrap_or(false) }

            /// Check symbolic equality with another quantity of the same type.
            pub fn equals(&self, other: &Self) -> bool { self.0.equals(&other.0).unwrap_or(false) }

            // ── Numerical evaluation ──────────────────────────────────

            /// Evaluate to f64.
            pub fn eval_f64(&self) -> Result<f64, crate::base::errors::SymplexError> { self.0.eval_f64() }

            /// Substitute integer values and evaluate to f64.
            pub fn eval_f64_with(&self, subs: &[(&Ex, i64)]) -> Result<f64, crate::base::errors::SymplexError> { self.0.eval_f64_with(subs) }

            /// Evaluate to a decimal string with the given number of digits.
            pub fn eval_decimal(&self, digits: u32) -> Result<String, crate::base::errors::SymplexError> { self.0.eval_decimal(digits) }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{} [{}]", self.0, $dim_sym)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }

        // --- Add: same type only ---

        impl ops::Add for $name {
            type Output = $name;
            fn add(self, rhs: $name) -> $name { $name(&self.0 + &rhs.0) }
        }
        impl ops::Add<&$name> for &$name {
            type Output = $name;
            fn add(self, rhs: &$name) -> $name { $name(&self.0 + &rhs.0) }
        }
        impl ops::Add<$name> for &$name {
            type Output = $name;
            fn add(self, rhs: $name) -> $name { $name(&self.0 + &rhs.0) }
        }
        impl ops::Add<&$name> for $name {
            type Output = $name;
            fn add(self, rhs: &$name) -> $name { $name(&self.0 + &rhs.0) }
        }

        // --- Sub: same type only ---

        impl ops::Sub for $name {
            type Output = $name;
            fn sub(self, rhs: $name) -> $name { $name(&self.0 - &rhs.0) }
        }
        impl ops::Sub<&$name> for &$name {
            type Output = $name;
            fn sub(self, rhs: &$name) -> $name { $name(&self.0 - &rhs.0) }
        }

        // --- Neg ---

        impl ops::Neg for $name {
            type Output = $name;
            fn neg(self) -> $name { $name(-&self.0) }
        }
        impl ops::Neg for &$name {
            type Output = $name;
            fn neg(self) -> $name { $name(-&self.0) }
        }

        // --- Scalar: type × i64, i64 × type ---

        impl ops::Mul<i64> for $name {
            type Output = $name;
            fn mul(self, rhs: i64) -> $name { $name(&self.0 * rhs) }
        }
        impl ops::Mul<i64> for &$name {
            type Output = $name;
            fn mul(self, rhs: i64) -> $name { $name(&self.0 * rhs) }
        }
        impl ops::Mul<$name> for i64 {
            type Output = $name;
            fn mul(self, rhs: $name) -> $name { $name(self * &rhs.0) }
        }
        impl ops::Mul<&$name> for i64 {
            type Output = $name;
            fn mul(self, rhs: &$name) -> $name { $name(self * &rhs.0) }
        }
        impl ops::Div<i64> for $name {
            type Output = $name;
            fn div(self, rhs: i64) -> $name { $name(&self.0 / rhs) }
        }

        // --- Scalar: type × &Ex, &Ex × type (dimensionless expression scaling) ---

        impl ops::Mul<&Ex> for $name {
            type Output = $name;
            fn mul(self, rhs: &Ex) -> $name { $name(&self.0 * rhs) }
        }
        impl ops::Mul<&Ex> for &$name {
            type Output = $name;
            fn mul(self, rhs: &Ex) -> $name { $name(&self.0 * rhs) }
        }
        impl ops::Mul<$name> for &Ex {
            type Output = $name;
            fn mul(self, rhs: $name) -> $name { $name(self * &rhs.0) }
        }

        // --- From/Into Qty<D> (named type → Qty is always safe) ---

        impl From<$name> for Qty<$dim> {
            fn from(q: $name) -> Qty<$dim> { Qty::from_ex(q.0) }
        }

        impl AsRef<$crate::prelude::Ex> for $name {
            fn as_ref(&self) -> &$crate::prelude::Ex { &self.0 }
        }
    };
}

// ===========================================================================
// 30 named quantity newtypes
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. Dimensionless / Angle (same underlying Dim — DimensionlessDim ≡ AngleDim)
// ---------------------------------------------------------------------------

define_quantity!(
    /// Dimensionless quantity (ratios, pure numbers).
    Dimensionless, DimensionlessDim, "Dimensionless", "1"
);

define_quantity!(
    @no_qty_from
    /// Angle \[rad\].
    Angle, AngleDim, "Angle", "rad"
);

// ---------------------------------------------------------------------------
// 2–7. Base SI quantities
// ---------------------------------------------------------------------------

define_quantity!(
    /// Length \[m\].
    Length, LengthDim, "Length", "m"
);

define_quantity!(
    /// Mass \[kg\].
    Mass, MassDim, "Mass", "kg"
);

define_quantity!(
    /// Time \[s\].
    Time, TimeDim, "Time", "s"
);

define_quantity!(
    /// Electric current \[A\].
    Current, CurrentDim, "Current", "A"
);

define_quantity!(
    /// Thermodynamic temperature \[K\].
    Temperature, TemperatureDim, "Temperature", "K"
);

// ---------------------------------------------------------------------------
// 8–9. Geometry
// ---------------------------------------------------------------------------

define_quantity!(
    /// Area \[m²\].
    Area, AreaDim, "Area", "m²"
);

define_quantity!(
    /// Volume \[m³\].
    Volume, VolumeDim, "Volume", "m³"
);

// ---------------------------------------------------------------------------
// 10–14. Kinematics
// ---------------------------------------------------------------------------

define_quantity!(
    /// Velocity \[m/s\].
    Velocity, VelocityDim, "Velocity", "m/s"
);

define_quantity!(
    /// Acceleration \[m/s²\].
    Acceleration, AccelerationDim, "Acceleration", "m/s²"
);

define_quantity!(
    /// Angular velocity \[rad/s\].
    AngularVelocity, AngularVelocityDim, "AngularVelocity", "rad/s"
);

define_quantity!(
    /// Angular acceleration \[rad/s²\].
    AngularAcceleration, AngularAccelerationDim, "AngularAcceleration", "rad/s²"
);

define_quantity!(
    @no_qty_from
    /// Frequency \[Hz\].
    Frequency, FrequencyDim, "Frequency", "Hz"
);

// ---------------------------------------------------------------------------
// 15–24. Mechanics
// ---------------------------------------------------------------------------

define_quantity!(
    /// Force \[N = kg·m/s²\].
    Force, ForceDim, "Force", "N"
);

define_quantity!(
    /// Energy \[J = kg·m²/s²\].
    Energy, EnergyDim, "Energy", "J"
);

define_quantity!(
    @no_qty_from
    /// Torque \[N·m\].
    Torque, TorqueDim, "Torque", "N·m"
);

define_quantity!(
    /// Power \[W = kg·m²/s³\].
    Power, PowerDim, "Power", "W"
);

define_quantity!(
    /// Momentum \[kg·m/s\].
    Momentum, MomentumDim, "Momentum", "kg·m/s"
);

define_quantity!(
    /// Angular momentum \[kg·m²/s\].
    AngularMomentum, AngularMomentumDim, "AngularMomentum", "kg·m²/s"
);

define_quantity!(
    /// Moment of inertia \[kg·m²\].
    MomentOfInertia, MomentOfInertiaDim, "MomentOfInertia", "kg·m²"
);

define_quantity!(
    /// Pressure \[Pa = kg/(m·s²)\].
    Pressure, PressureDim, "Pressure", "Pa"
);

define_quantity!(
    /// Spring stiffness \[N/m\].
    Stiffness, StiffnessDim, "Stiffness", "N/m"
);

define_quantity!(
    /// Viscous damping coefficient \[N·s/m\].
    Damping, DampingDim, "Damping", "N·s/m"
);

// ---------------------------------------------------------------------------
// 25–30. Electromagnetism
// ---------------------------------------------------------------------------

define_quantity!(
    /// Electric potential \[V\].
    Voltage, VoltageDim, "Voltage", "V"
);

define_quantity!(
    /// Electrical resistance \[Ω\].
    Resistance, ResistanceDim, "Resistance", "Ω"
);

define_quantity!(
    /// Inductance \[H\].
    Inductance, InductanceDim, "Inductance", "H"
);

define_quantity!(
    /// Capacitance \[F\].
    Capacitance, CapacitanceDim, "Capacitance", "F"
);

define_quantity!(
    /// Electric charge \[C\].
    Charge, ChargeDim, "Charge", "C"
);

define_quantity!(
    /// Magnetic flux \[Wb\].
    MagneticFlux, MagneticFluxDim, "MagneticFlux", "Wb"
);

// ===========================================================================
// Trig functions — Angle → Dimensionless only
// ===========================================================================

impl Angle {
    /// Sine of this angle, returning a dimensionless result.
    pub fn sin(&self) -> Dimensionless { Dimensionless(self.0.sin()) }

    /// Cosine of this angle, returning a dimensionless result.
    pub fn cos(&self) -> Dimensionless { Dimensionless(self.0.cos()) }

    /// Tangent of this angle, returning a dimensionless result.
    pub fn tan(&self) -> Dimensionless { Dimensionless(self.0.tan()) }
}

// ===========================================================================
// Cross-type From impls for dimension-collision pairs
// ===========================================================================

// Energy ↔ Torque (both Dim<P2,P1,N2,Z0,Z0,Z0,Z0>)
impl From<Energy> for Torque {
    fn from(e: Energy) -> Self { Torque(e.0) }
}
impl From<Torque> for Energy {
    fn from(t: Torque) -> Self { Energy(t.0) }
}

// Frequency ↔ AngularVelocity (both Dim<Z0,Z0,N1,Z0,Z0,Z0,Z0>)
impl From<Frequency> for AngularVelocity {
    fn from(f: Frequency) -> Self { AngularVelocity(f.0) }
}
impl From<AngularVelocity> for Frequency {
    fn from(w: AngularVelocity) -> Self { Frequency(w.0) }
}

// ===========================================================================
// Named conversion methods for types that lack From<Qty<D>>
// ===========================================================================

impl Angle {
    /// Convert from a `Dimensionless` value.
    ///
    /// `Angle` and `Dimensionless` share the same dimension vector, so the
    /// generic `From<Qty<DimensionlessDim>>` impl is given to `Dimensionless`.
    /// Use this named method instead.
    pub fn from_dimensionless(d: Dimensionless) -> Angle {
        Angle(d.0)
    }
}

impl Torque {
    /// Convert from an `Energy` value.
    ///
    /// `Torque` and `Energy` share the same dimension vector, so the
    /// generic `From<Qty<EnergyDim>>` impl is given to `Energy`.
    /// Use this named method instead.
    pub fn from_energy(e: Energy) -> Torque {
        Torque(e.0)
    }
}

impl Frequency {
    /// Convert from an `AngularVelocity` value.
    ///
    /// `Frequency` and `AngularVelocity` share the same dimension vector, so
    /// the generic `From<Qty<AngularVelocityDim>>` impl is given to
    /// `AngularVelocity`. Use this named method instead.
    pub fn from_angular_velocity(w: AngularVelocity) -> Frequency {
        Frequency(w.0)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_same_type() {
        let a = Force::symbol("F1");
        let b = Force::symbol("F2");
        let c = &a + &b;
        assert_eq!(Force::dim_name_str(), "Force");
        assert_eq!(Force::dim_symbol_str(), "N");
        assert!(format!("{}", c).contains("[N]"));
    }

    #[test]
    fn sub_same_type() {
        let a = Length::constant(10);
        let b = Length::constant(3);
        let c = a - b;
        assert!(format!("{}", c).contains("[m]"));
    }

    #[test]
    fn neg() {
        let v = Velocity::symbol("v");
        let neg_v = -&v;
        assert!(format!("{:?}", neg_v).starts_with("Velocity("));
    }

    #[test]
    fn scalar_mul_i64() {
        let m = Mass::symbol("m");
        let double = &m * 2;
        assert!(format!("{}", double).contains("[kg]"));

        let triple = 3 * &m;
        assert!(format!("{}", triple).contains("[kg]"));
    }

    #[test]
    fn scalar_div_i64() {
        let e = Energy::constant(100);
        let half = e / 2;
        assert!(format!("{}", half).contains("[J]"));
    }

    #[test]
    fn scalar_mul_ex() {
        let f = Force::symbol("F");
        let k = crate::default_context().symbol("k");
        let scaled = &f * &k;
        assert!(format!("{}", scaled).contains("[N]"));
    }

    #[test]
    fn display_and_debug() {
        let t = Temperature::symbol("T");
        assert!(format!("{}", t).contains("[K]"));
        assert!(format!("{:?}", t).starts_with("Temperature("));
    }

    #[test]
    fn angle_trig() {
        let theta = Angle::symbol("theta");
        let s = theta.sin();
        assert_eq!(Dimensionless::dim_symbol_str(), "1");
        assert!(format!("{}", s).contains("[1]"));
    }

    #[test]
    fn cross_type_from_energy_torque() {
        let e = Energy::symbol("E");
        let t: Torque = Torque::from(e);
        assert!(format!("{}", t).contains("[N·m]"));

        let t2 = Torque::symbol("tau");
        let e2: Energy = Energy::from(t2);
        assert!(format!("{}", e2).contains("[J]"));
    }

    #[test]
    fn cross_type_from_freq_angular_vel() {
        let w = AngularVelocity::symbol("omega");
        let f: Frequency = Frequency::from(w);
        assert!(format!("{}", f).contains("[Hz]"));
    }

    #[test]
    fn named_conversion_angle_from_dimensionless() {
        let d = Dimensionless::constant(1);
        let a = Angle::from_dimensionless(d);
        assert!(format!("{}", a).contains("[rad]"));
    }

    #[test]
    fn named_conversion_torque_from_energy() {
        let e = Energy::symbol("W");
        let t = Torque::from_energy(e);
        assert_eq!(Torque::dim_name_str(), "Torque");
        assert!(format!("{}", t).contains("[N·m]"));
    }

    #[test]
    fn named_conversion_frequency_from_angular_velocity() {
        let w = AngularVelocity::symbol("omega");
        let f = Frequency::from_angular_velocity(w);
        assert_eq!(Frequency::dim_name_str(), "Frequency");
        assert!(format!("{}", f).contains("[Hz]"));
    }

    #[test]
    fn into_inner_roundtrip() {
        let raw = crate::default_context().symbol("x");
        let q = Pressure::from_ex(raw.clone());
        let back = q.into_inner();
        assert_eq!(format!("{}", back), format!("{}", raw));
    }

    #[test]
    fn zero_and_constant() {
        let z = Charge::zero();
        assert!(format!("{}", z).contains("[C]"));
        let c = Charge::constant(42);
        assert!(format!("{}", c).contains("[C]"));
    }

    #[test]
    fn qty_roundtrip() {
        let v = Voltage::symbol("V");
        let q: Qty<VoltageDim> = v.into();
        let v2: Voltage = q.into();
        assert!(format!("{}", v2).contains("[V]"));
    }

    #[test]
    fn simplify_expand_eval() {
        let x = Length::symbol("x");
        let _ = x.clone().simplify();
        let _ = x.clone().expand();
        let _ = x.eval();
    }

    #[test]
    fn subs() {
        let x_var = crate::default_context().symbol("x");
        let val = crate::default_context().int(5);
        let len = Length::symbol("x");
        let result = len.subs(&x_var, &val);
        assert!(format!("{}", result).contains("[m]"));
    }
}
