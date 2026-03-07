use std::ops;
use super::si::*;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Macros
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

macro_rules! impl_named_mul {
    ($Lhs:ty, $Rhs:ty => $Out:ty) => {
        impl ops::Mul<$Rhs> for $Lhs {
            type Output = $Out;
            fn mul(self, rhs: $Rhs) -> $Out { <$Out>::from_ex(&self.0 * &rhs.0) }
        }
        impl ops::Mul<&$Rhs> for &$Lhs {
            type Output = $Out;
            fn mul(self, rhs: &$Rhs) -> $Out { <$Out>::from_ex(&self.0 * &rhs.0) }
        }
        impl ops::Mul<$Rhs> for &$Lhs {
            type Output = $Out;
            fn mul(self, rhs: $Rhs) -> $Out { <$Out>::from_ex(&self.0 * &rhs.0) }
        }
        impl ops::Mul<&$Rhs> for $Lhs {
            type Output = $Out;
            fn mul(self, rhs: &$Rhs) -> $Out { <$Out>::from_ex(&self.0 * &rhs.0) }
        }
    };
}

macro_rules! impl_named_div {
    ($Lhs:ty, $Rhs:ty => $Out:ty) => {
        impl ops::Div<$Rhs> for $Lhs {
            type Output = $Out;
            fn div(self, rhs: $Rhs) -> $Out { <$Out>::from_ex(&self.0 / &rhs.0) }
        }
        impl ops::Div<&$Rhs> for &$Lhs {
            type Output = $Out;
            fn div(self, rhs: &$Rhs) -> $Out { <$Out>::from_ex(&self.0 / &rhs.0) }
        }
        impl ops::Div<$Rhs> for &$Lhs {
            type Output = $Out;
            fn div(self, rhs: $Rhs) -> $Out { <$Out>::from_ex(&self.0 / &rhs.0) }
        }
        impl ops::Div<&$Rhs> for $Lhs {
            type Output = $Out;
            fn div(self, rhs: &$Rhs) -> $Out { <$Out>::from_ex(&self.0 / &rhs.0) }
        }
    };
}

macro_rules! impl_dimensionless_mul {
    ($Qty:ty) => {
        impl ops::Mul<Dimensionless> for $Qty {
            type Output = $Qty;
            fn mul(self, rhs: Dimensionless) -> $Qty { <$Qty>::from_ex(&self.0 * &rhs.0) }
        }
        impl ops::Mul<$Qty> for Dimensionless {
            type Output = $Qty;
            fn mul(self, rhs: $Qty) -> $Qty { <$Qty>::from_ex(&self.0 * &rhs.0) }
        }
        impl ops::Mul<&Dimensionless> for &$Qty {
            type Output = $Qty;
            fn mul(self, rhs: &Dimensionless) -> $Qty { <$Qty>::from_ex(&self.0 * &rhs.0) }
        }
        impl ops::Mul<&$Qty> for &Dimensionless {
            type Output = $Qty;
            fn mul(self, rhs: &$Qty) -> $Qty { <$Qty>::from_ex(&self.0 * &rhs.0) }
        }
    };
}

macro_rules! impl_angle_mul {
    ($Qty:ty) => {
        impl ops::Mul<Angle> for $Qty {
            type Output = $Qty;
            fn mul(self, rhs: Angle) -> $Qty { <$Qty>::from_ex(&self.0 * &rhs.0) }
        }
        impl ops::Mul<$Qty> for Angle {
            type Output = $Qty;
            fn mul(self, rhs: $Qty) -> $Qty { <$Qty>::from_ex(&self.0 * &rhs.0) }
        }
        impl ops::Mul<&Angle> for &$Qty {
            type Output = $Qty;
            fn mul(self, rhs: &Angle) -> $Qty { <$Qty>::from_ex(&self.0 * &rhs.0) }
        }
        impl ops::Mul<&$Qty> for &Angle {
            type Output = $Qty;
            fn mul(self, rhs: &$Qty) -> $Qty { <$Qty>::from_ex(&self.0 * &rhs.0) }
        }
    };
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Multiplication rules  (each invocation generates 4 impls)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ── Mechanics ──────────────────────────────────────────────────────────────
impl_named_mul!(Mass, Acceleration => Force);
impl_named_mul!(Acceleration, Mass => Force);
impl_named_mul!(Force, Length => Energy);
impl_named_mul!(Length, Force => Energy);
impl_named_mul!(Mass, Velocity => Momentum);
impl_named_mul!(Velocity, Mass => Momentum);
impl_named_mul!(Force, Velocity => Power);
impl_named_mul!(Velocity, Force => Power);
impl_named_mul!(Acceleration, Time => Velocity);
impl_named_mul!(Time, Acceleration => Velocity);
impl_named_mul!(Velocity, Time => Length);
impl_named_mul!(Time, Velocity => Length);
impl_named_mul!(Length, Length => Area);
impl_named_mul!(Length, Area => Volume);
impl_named_mul!(Area, Length => Volume);
impl_named_mul!(Mass, Area => MomentOfInertia);
impl_named_mul!(Area, Mass => MomentOfInertia);
impl_named_mul!(Stiffness, Length => Force);
impl_named_mul!(Length, Stiffness => Force);
impl_named_mul!(Damping, Velocity => Force);
impl_named_mul!(Velocity, Damping => Force);

// ── Rotational ─────────────────────────────────────────────────────────────
impl_named_mul!(MomentOfInertia, AngularAcceleration => Torque);
impl_named_mul!(AngularAcceleration, MomentOfInertia => Torque);
impl_named_mul!(MomentOfInertia, AngularVelocity => AngularMomentum);
impl_named_mul!(AngularVelocity, MomentOfInertia => AngularMomentum);
impl_named_mul!(AngularVelocity, Time => Angle);
impl_named_mul!(Time, AngularVelocity => Angle);

// ── Electrical ─────────────────────────────────────────────────────────────
impl_named_mul!(Current, Resistance => Voltage);
impl_named_mul!(Resistance, Current => Voltage);
impl_named_mul!(Voltage, Current => Power);
impl_named_mul!(Current, Voltage => Power);
impl_named_mul!(Current, Time => Charge);
impl_named_mul!(Time, Current => Charge);
impl_named_mul!(Inductance, Current => MagneticFlux);
impl_named_mul!(Current, Inductance => MagneticFlux);
impl_named_mul!(Charge, Voltage => Energy);
impl_named_mul!(Voltage, Charge => Energy);
impl_named_mul!(MagneticFlux, AngularVelocity => Voltage);
impl_named_mul!(AngularVelocity, MagneticFlux => Voltage);

// ── Power / Energy ─────────────────────────────────────────────────────────
impl_named_mul!(Power, Time => Energy);
impl_named_mul!(Time, Power => Energy);
impl_named_mul!(Energy, Frequency => Power);
impl_named_mul!(Frequency, Energy => Power);

// ── Dimensionless × Dimensionless ──────────────────────────────────────────
impl ops::Mul for Dimensionless {
    type Output = Dimensionless;
    fn mul(self, rhs: Dimensionless) -> Dimensionless { Dimensionless::from_ex(&self.0 * &rhs.0) }
}
impl ops::Mul<&Dimensionless> for &Dimensionless {
    type Output = Dimensionless;
    fn mul(self, rhs: &Dimensionless) -> Dimensionless { Dimensionless::from_ex(&self.0 * &rhs.0) }
}

// ── Angle × Angle ──────────────────────────────────────────────────────────
impl ops::Mul for Angle {
    type Output = Angle;
    fn mul(self, rhs: Angle) -> Angle { Angle::from_ex(&self.0 * &rhs.0) }
}
impl ops::Mul<&Angle> for &Angle {
    type Output = Angle;
    fn mul(self, rhs: &Angle) -> Angle { Angle::from_ex(&self.0 * &rhs.0) }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Division rules  (each invocation generates 4 impls)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ── Mechanics ──────────────────────────────────────────────────────────────
impl_named_div!(Force, Mass => Acceleration);
impl_named_div!(Force, Acceleration => Mass);
impl_named_div!(Length, Time => Velocity);
impl_named_div!(Velocity, Time => Acceleration);
impl_named_div!(Energy, Time => Power);
impl_named_div!(Energy, Power => Time);
impl_named_div!(Energy, Length => Force);
impl_named_div!(Energy, Force => Length);
impl_named_div!(Momentum, Mass => Velocity);
impl_named_div!(Momentum, Time => Force);
impl_named_div!(Momentum, Velocity => Mass);
impl_named_div!(Power, Velocity => Force);
impl_named_div!(Power, Force => Velocity);
impl_named_div!(Force, Stiffness => Length);
impl_named_div!(Force, Damping => Velocity);

// ── Rotational ─────────────────────────────────────────────────────────────
impl_named_div!(AngularMomentum, MomentOfInertia => AngularVelocity);
impl_named_div!(Torque, MomentOfInertia => AngularAcceleration);
impl_named_div!(Angle, Time => AngularVelocity);

// ── Electrical ─────────────────────────────────────────────────────────────
impl_named_div!(Voltage, Current => Resistance);
impl_named_div!(Voltage, Resistance => Current);
impl_named_div!(Power, Current => Voltage);
impl_named_div!(Power, Voltage => Current);
impl_named_div!(Charge, Time => Current);
impl_named_div!(Charge, Current => Time);
impl_named_div!(MagneticFlux, Current => Inductance);
impl_named_div!(MagneticFlux, Inductance => Current);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Dimensionless scaling  (Dimensionless × Q = Q, Q × Dimensionless = Q)
//
// 28 non-Dimensionless, non-Angle types, plus Angle itself at the end.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl_dimensionless_mul!(Length);
impl_dimensionless_mul!(Mass);
impl_dimensionless_mul!(Time);
impl_dimensionless_mul!(Velocity);
impl_dimensionless_mul!(Acceleration);
impl_dimensionless_mul!(Force);
impl_dimensionless_mul!(Energy);
impl_dimensionless_mul!(Power);
impl_dimensionless_mul!(Momentum);
impl_dimensionless_mul!(MomentOfInertia);
impl_dimensionless_mul!(Torque);
impl_dimensionless_mul!(AngularVelocity);
impl_dimensionless_mul!(AngularAcceleration);
impl_dimensionless_mul!(AngularMomentum);
impl_dimensionless_mul!(Frequency);
impl_dimensionless_mul!(Area);
impl_dimensionless_mul!(Volume);
impl_dimensionless_mul!(Voltage);
impl_dimensionless_mul!(Resistance);
impl_dimensionless_mul!(Current);
impl_dimensionless_mul!(Inductance);
impl_dimensionless_mul!(Capacitance);
impl_dimensionless_mul!(Charge);
impl_dimensionless_mul!(MagneticFlux);
impl_dimensionless_mul!(Pressure);
impl_dimensionless_mul!(Stiffness);
impl_dimensionless_mul!(Damping);
impl_dimensionless_mul!(Temperature);

// Dimensionless × Angle  and  Angle × Dimensionless
impl_dimensionless_mul!(Angle);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Angle scaling  (Angle acts as dimensionless for multiplication)
//
// Angle × Q = Q, Q × Angle = Q  for the types where it is physically
// meaningful.  NOTE: Dimensionless is NOT listed here — that pairing is
// already covered above by `impl_dimensionless_mul!(Angle)`.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl_angle_mul!(Length);
impl_angle_mul!(Mass);
impl_angle_mul!(Time);
impl_angle_mul!(Velocity);
impl_angle_mul!(Acceleration);
impl_angle_mul!(Force);
impl_angle_mul!(Energy);
impl_angle_mul!(Torque);
impl_angle_mul!(Power);
impl_angle_mul!(Momentum);
impl_angle_mul!(MomentOfInertia);
impl_angle_mul!(Voltage);
impl_angle_mul!(Resistance);
impl_angle_mul!(Current);
impl_angle_mul!(Inductance);
impl_angle_mul!(Capacitance);
impl_angle_mul!(Charge);
impl_angle_mul!(MagneticFlux);
impl_angle_mul!(Pressure);
impl_angle_mul!(Stiffness);
impl_angle_mul!(Damping);
impl_angle_mul!(AngularMomentum);
impl_angle_mul!(AngularVelocity);
impl_angle_mul!(AngularAcceleration);
impl_angle_mul!(Frequency);
impl_angle_mul!(Area);
impl_angle_mul!(Volume);
impl_angle_mul!(Temperature);
