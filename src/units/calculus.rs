//! Dimensional calculus: differentiation and integration that track
//! physical dimensions through the type system.
//!
//! Differentiation divides dimensions: d(Qty<D1>)/d(Qty<D2>) → Qty<D1/D2>
//! Integration multiplies dimensions: ∫ Qty<D1> d(Qty<D2>) → Qty<D1*D2>
//!
//! Named newtypes use `.as_qty()` before calling these functions,
//! then convert back via `.into()`:
//!
//! ```ignore
//! let x = Length::symbol("x");
//! let t = Time::symbol("t");
//! let v: Velocity = diff_qty(&x.as_qty(), &t.as_qty()).into();
//! ```

use std::ops;

use typenum::operator_aliases::{Diff, Sum};

use crate::prelude::Ex;

use super::dim::Dim;
use super::qty::Qty;

// ═══════════════════════════════════════════════════════════════════════════
// diff_qty — symbolic differentiation with dimension tracking
// ═══════════════════════════════════════════════════════════════════════════

/// Differentiate a dimensioned expression with respect to a dimensioned variable.
///
/// The output dimension is the quotient of the two input dimensions:
/// `D_expr / D_var` (exponents are subtracted).
///
/// # Type-level mechanics
///
/// Given `expr: Qty<Dim<L1, M1, T1, …>>` and `var: Qty<Dim<L2, M2, T2, …>>`,
/// the result has dimension `Dim<L1−L2, M1−M2, T1−T2, …>`.
///
/// # Examples
///
/// ```ignore
/// use symplex::units::*;
///
/// let x: Qty<LengthDim> = Qty::from_ex(symplex::var("x"));
/// let t: Qty<TimeDim>   = Qty::from_ex(symplex::var("t"));
/// let v: Qty<VelocityDim> = diff_qty(&x, &t);   // Length / Time = Velocity
/// ```
pub fn diff_qty<L1, M1, T1, I1, Th1, N1x, J1,
                L2, M2, T2, I2, Th2, N2x, J2>(
    expr: &Qty<Dim<L1, M1, T1, I1, Th1, N1x, J1>>,
    var:  &Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>,
) -> Qty<Dim<Diff<L1, L2>, Diff<M1, M2>, Diff<T1, T2>,
            Diff<I1, I2>, Diff<Th1, Th2>, Diff<N1x, N2x>, Diff<J1, J2>>>
where
    L1:  ops::Sub<L2>,
    M1:  ops::Sub<M2>,
    T1:  ops::Sub<T2>,
    I1:  ops::Sub<I2>,
    Th1: ops::Sub<Th2>,
    N1x: ops::Sub<N2x>,
    J1:  ops::Sub<J2>,
{
    Qty::from_ex(expr.inner().diff(var.inner()))
}

// ═══════════════════════════════════════════════════════════════════════════
// integrate_qty — symbolic integration with dimension tracking
// ═══════════════════════════════════════════════════════════════════════════

/// Integrate a dimensioned expression with respect to a dimensioned variable.
///
/// The output dimension is the product of the two input dimensions:
/// `D_expr × D_var` (exponents are added).
///
/// # Type-level mechanics
///
/// Given `expr: Qty<Dim<L1, M1, T1, …>>` and `var: Qty<Dim<L2, M2, T2, …>>`,
/// the result has dimension `Dim<L1+L2, M1+M2, T1+T2, …>`.
///
/// # Examples
///
/// ```ignore
/// use symplex::units::*;
///
/// let f: Qty<ForceDim>  = Qty::from_ex(symplex::var("F"));
/// let x: Qty<LengthDim> = Qty::from_ex(symplex::var("x"));
/// let w: Qty<EnergyDim> = integrate_qty(&f, &x);  // Force × Length = Energy
/// ```
pub fn integrate_qty<L1, M1, T1, I1, Th1, N1x, J1,
                     L2, M2, T2, I2, Th2, N2x, J2>(
    expr: &Qty<Dim<L1, M1, T1, I1, Th1, N1x, J1>>,
    var:  &Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>,
) -> Qty<Dim<Sum<L1, L2>, Sum<M1, M2>, Sum<T1, T2>,
            Sum<I1, I2>, Sum<Th1, Th2>, Sum<N1x, N2x>, Sum<J1, J2>>>
where
    L1:  ops::Add<L2>,
    M1:  ops::Add<M2>,
    T1:  ops::Add<T2>,
    I1:  ops::Add<I2>,
    Th1: ops::Add<Th2>,
    N1x: ops::Add<N2x>,
    J1:  ops::Add<J2>,
{
    Qty::from_ex(expr.inner().integrate(var.inner()))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::dim::*;
    use super::super::si::*;

    // --- Differentiation tests ---

    /// d(Length)/d(Time) = Velocity
    #[test]
    fn diff_length_by_time_is_velocity() {
        let x: Qty<LengthDim> = Qty::from_ex(crate::var("x"));
        let t: Qty<TimeDim>   = Qty::from_ex(crate::var("t"));

        let result = diff_qty(&x, &t);

        // If dimensions are wrong, this `.into()` won't compile.
        let _v: Velocity = result.into();
    }

    /// d(Velocity)/d(Time) = Acceleration
    #[test]
    fn diff_velocity_by_time_is_acceleration() {
        let v: Qty<VelocityDim> = Qty::from_ex(crate::var("v"));
        let t: Qty<TimeDim>     = Qty::from_ex(crate::var("t"));

        let result = diff_qty(&v, &t);

        let _a: Acceleration = result.into();
    }

    /// d(Energy)/d(Length) = Force
    #[test]
    fn diff_energy_by_length_is_force() {
        let e: Qty<EnergyDim>  = Qty::from_ex(crate::var("E"));
        let x: Qty<LengthDim>  = Qty::from_ex(crate::var("x"));

        let result = diff_qty(&e, &x);

        let _f: Force = result.into();
    }

    // --- Integration tests ---

    /// ∫ Force d(Length) = Energy
    #[test]
    fn integrate_force_over_length_is_energy() {
        let f: Qty<ForceDim>   = Qty::from_ex(crate::var("F"));
        let x: Qty<LengthDim>  = Qty::from_ex(crate::var("x"));

        let result = integrate_qty(&f, &x);

        let _w: Energy = result.into();
    }

    /// ∫ Velocity d(Time) = Length
    #[test]
    fn integrate_velocity_over_time_is_length() {
        let v: Qty<VelocityDim> = Qty::from_ex(crate::var("v"));
        let t: Qty<TimeDim>     = Qty::from_ex(crate::var("t"));

        let result = integrate_qty(&v, &t);

        let _x: Length = result.into();
    }

    // --- Fundamental Theorem of Calculus roundtrip ---

    /// diff(integrate(v, t), t) must have Velocity dimension.
    ///
    /// This verifies that the type-level arithmetic is consistent:
    /// Velocity×Time = Length, then Length/Time = Velocity.
    #[test]
    fn ftc_roundtrip_velocity() {
        let v: Qty<VelocityDim> = Qty::from_ex(crate::var("v"));
        let t: Qty<TimeDim>     = Qty::from_ex(crate::var("t"));

        // integrate: Velocity × Time = Length
        let integrated = integrate_qty(&v, &t);

        // differentiate: Length / Time = Velocity
        let t2: Qty<TimeDim> = Qty::from_ex(crate::var("t"));
        let roundtrip = diff_qty(&integrated, &t2);

        // Must compile as Velocity — proves FTC dimensional consistency.
        let _v2: Velocity = roundtrip.into();
    }

    // --- Named-type workflow via as_qty ---

    /// Demonstrate the full named → Qty → calculus → named workflow.
    #[test]
    fn named_type_workflow() {
        let x = Length::symbol("x");
        let t = Time::symbol("t");

        let v: Velocity = diff_qty(&x.as_qty(), &t.as_qty()).into();

        // The result is a proper Velocity; we can use named-type methods.
        assert_eq!(Velocity::dim_name_str(), "Velocity");
        assert_eq!(Velocity::dim_symbol_str(), "m/s");
        let _ = v;
    }
}
