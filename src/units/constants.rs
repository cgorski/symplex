//! Physical constants as dimension-typed symbolic expressions.
//!
//! Each constant displays as its standard symbol (e.g., "c" for speed of light)
//! and evaluates to its exact value when numerical evaluation is requested.
//! All values are exact — defined by the 2019 SI redefinition or by international agreement.
//!
//! # Examples
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::units::*;
//! use symplex::units::constants;
//!
//! let ctx = symplex::api::context::Context::new();
//! let c = constants::speed_of_light(&ctx);
//! let m = Mass::symbol(&ctx, "m");
//! let energy = Energy::from_ex(m.inner() * c.inner() * c.inner());  // E = mc²
//! // Displays as "c^2*m", not "89875517873681764*m"
//! ```

use super::si::*;
use super::qty::Qty;
use super::dim::*;
use typenum::{N1, N2, P1, P2, P3, Z0};

/// Speed of light in vacuum: c = 299,792,458 m/s (exact since 2019 SI redefinition).
pub fn speed_of_light(ctx: &crate::api::context::Context) -> Velocity {
    Velocity::from_ex(ctx.physical_constant("c", ctx.int(299_792_458)))
}

/// Standard acceleration of gravity: g₀ = 9.80665 m/s² (exact by definition, 1901).
pub fn standard_gravity(ctx: &crate::api::context::Context) -> Acceleration {
    Acceleration::from_ex(ctx.physical_constant("g_0", ctx.rational(980665, 100_000)))
}

/// Elementary charge: e = 1.602176634 × 10⁻¹⁹ C (exact since 2019 SI redefinition).
pub fn elementary_charge(ctx: &crate::api::context::Context) -> Charge {
    Charge::from_ex(ctx.physical_constant("e_0", &ctx.int(1_602_176_634) / &ctx.int(10).powi(28)))
}

/// Planck constant: h = 6.62607015 × 10⁻³⁴ J·s (exact since 2019 SI redefinition).
///
/// Returns as `AngularMomentum` (dimension M·L²·T⁻¹, same as Action = Energy × Time).
pub fn planck_constant(ctx: &crate::api::context::Context) -> AngularMomentum {
    AngularMomentum::from_ex(ctx.physical_constant("h", &ctx.int(662_607_015) / &ctx.int(10).powi(42)))
}

/// Reduced Planck constant: ℏ = h/(2π) (exact).
///
/// Note: This involves π, so the value is symbolic: h/(2π).
/// For numerical evaluation, both h and π resolve to exact values.
pub fn reduced_planck_constant(ctx: &crate::api::context::Context) -> AngularMomentum {
    let h = ctx.physical_constant("h", &ctx.int(662_607_015) / &ctx.int(10).powi(42));
    let two_pi = &(ctx.int(2) * &ctx.pi());
    AngularMomentum::from_ex(ctx.physical_constant("hbar", &h / two_pi))
}

/// Boltzmann constant: k_B = 1.380649 × 10⁻²³ J/K (exact since 2019 SI redefinition).
///
/// Dimension: M·L²·T⁻²·Θ⁻¹ (Energy per Temperature).
/// Returns as `Qty` since there's no named type for this dimension.
pub fn boltzmann_constant(ctx: &crate::api::context::Context) -> Qty<Dim<P2, P1, N2, Z0, N1, Z0, Z0>> {
    Qty::from_ex(ctx.physical_constant("k_B", &ctx.int(1_380_649) / &ctx.int(10).powi(29)))
}

/// Avogadro constant: N_A = 6.02214076 × 10²³ mol⁻¹ (exact since 2019 SI redefinition).
///
/// Dimension: N⁻¹ (inverse amount of substance).
/// Returns as `Qty` since there's no named type for this dimension.
pub fn avogadro_constant(ctx: &crate::api::context::Context) -> Qty<Dim<Z0, Z0, Z0, Z0, Z0, N1, Z0>> {
    // N_A = 602214076 × 10^15
    let val = &ctx.int(602_214_076) * &ctx.int(10).powi(15);
    Qty::from_ex(ctx.physical_constant("N_A", val))
}

/// Newtonian gravitational constant: G ≈ 6.67430 × 10⁻¹¹ m³/(kg·s²).
///
/// NOTE: Unlike the other constants here, G is NOT exact — it is measured experimentally.
/// The value 6.67430e-11 is the 2018 CODATA recommended value.
/// Dimension: L³·M⁻¹·T⁻²
pub fn gravitational_constant(ctx: &crate::api::context::Context) -> Qty<Dim<P3, N1, N2, Z0, Z0, Z0, Z0>> {
    Qty::from_ex(ctx.physical_constant("G", &ctx.int(667_430) / &ctx.int(10).powi(16)))
}

/// Pre-built dimension map containing all physical constants.
///
/// Use with `infer_dimension` for runtime dimension checking of expressions
/// containing physical constants.
pub fn physical_constants_dimmap() -> super::inference::DimMap {
    use super::inference::DimMap;
    DimMap::new()
        .with("c", ConstDim::VELOCITY)
        .with("g_0", ConstDim::ACCELERATION)
        .with("e_0", ConstDim::CHARGE)
        .with("h", ConstDim::ANGULAR_MOMENTUM)
        .with("hbar", ConstDim::ANGULAR_MOMENTUM)
        .with("k_B", ConstDim::new(2, 1, -2, 0, -1, 0, 0))
        .with("N_A", ConstDim::new(0, 0, 0, 0, 0, -1, 0))
        .with("G", ConstDim::new(3, -1, -2, 0, 0, 0, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_of_light_is_velocity() {
        let ctx = crate::api::context::Context::new();
        let c = speed_of_light(&ctx);
        assert!(format!("{}", c.inner()).contains("c"), "should display as c");
    }

    #[test]
    fn speed_of_light_eval_f64() {
        let ctx = crate::api::context::Context::new();
        let c = speed_of_light(&ctx);
        let val = c.eval_f64().unwrap();
        assert!((val - 299_792_458.0).abs() < 1.0, "c = {val}");
    }

    #[test]
    fn elementary_charge_eval() {
        let ctx = crate::api::context::Context::new();
        let e = elementary_charge(&ctx);
        let val = e.eval_f64().unwrap();
        assert!((val - 1.602176634e-19).abs() / 1.602176634e-19 < 1e-10,
            "e = {val}");
    }

    #[test]
    fn planck_constant_eval() {
        let ctx = crate::api::context::Context::new();
        let h = planck_constant(&ctx);
        let val = h.eval_f64().unwrap();
        assert!((val - 6.62607015e-34).abs() / 6.62607015e-34 < 1e-10,
            "h = {val}");
    }

    #[test]
    fn boltzmann_eval() {
        let ctx = crate::api::context::Context::new();
        let kb = boltzmann_constant(&ctx);
        let val = kb.eval_f64().unwrap();
        assert!((val - 1.380649e-23).abs() / 1.380649e-23 < 1e-10,
            "k_B = {val}");
    }

    #[test]
    fn standard_gravity_eval() {
        let ctx = crate::api::context::Context::new();
        let g = standard_gravity(&ctx);
        let val = g.eval_f64().unwrap();
        assert!((val - 9.80665).abs() < 1e-10, "g = {val}");
    }

    #[test]
    fn e_equals_mc_squared() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; m);
        let c = speed_of_light(&ctx);
        let mass = Mass::symbol(&ctx, "m");
        // Use raw expression arithmetic to avoid missing named-mul impls
        let energy = Energy::from_ex(mass.inner() * c.inner() * c.inner());
        // Display should contain "c", not the numeric value
        let display = format!("{}", energy.inner());
        assert!(display.contains("c"), "E=mc² should display symbolically: {display}");
        // Evaluate with m=1 kg
        let val = energy.subs(&m, &ctx.int(1)).eval_f64().unwrap();
        let expected = 299_792_458.0_f64 * 299_792_458.0;
        assert!((val - expected).abs() / expected < 1e-10,
            "E(m=1) = {val}, expected {expected}");
    }

    #[test]
    fn constant_diff_is_zero() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; x);
        let c = speed_of_light(&ctx);
        let dc_dx = c.inner().diff(&x);
        assert!(dc_dx.is_zero().unwrap_or(false) || format!("{}", dc_dx) == "0",
            "d/dx(c) should be 0, got {dc_dx}");
    }

    #[test]
    fn constant_in_product_diff() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; x);
        let c = speed_of_light(&ctx);
        let cx = c.inner() * &x;
        let d = cx.diff(&x);
        // d/dx(c*x) = c
        let display = format!("{}", d);
        assert!(display.contains("c"), "d/dx(c*x) should contain c: {display}");
    }

    #[test]
    fn constant_survives_simplify() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; x, y);
        let c = speed_of_light(&ctx);
        let expr = c.inner() * &x + c.inner() * &y;
        let simplified = expr.simplify();
        let display = format!("{}", simplified);
        assert!(display.contains("c"), "simplify should preserve c: {display}");
    }

    #[test]
    fn physical_constants_dimmap_works() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; m);
        let dims = physical_constants_dimmap()
            .with("m", ConstDim::MASS);
        let c = speed_of_light(&ctx);
        let mc2 = &(m.clone() * c.inner()) * c.inner();
        let dim = crate::units::inference::infer_dimension(&mc2, &dims);
        assert!(dim.is_ok(), "should infer dimension of mc²: {:?}", dim);
        assert!(dim.unwrap().eq(ConstDim::ENERGY),
            "mc² should have Energy dimension");
    }
}
