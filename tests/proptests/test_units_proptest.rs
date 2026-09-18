//! Property-based tests for dimension invariants.

use proptest::prelude::*;
use symplex::units::*;

use symplex::prelude::*;
/// Generate a DimMap for common physics variables.
fn physics_dims() -> DimMap {
    DimMap::new()
        .with("m", ConstDim::MASS)
        .with("a", ConstDim::ACCELERATION)
        .with("v", ConstDim::VELOCITY)
        .with("t", ConstDim::TIME)
        .with("x", ConstDim::LENGTH)
        .with("k", ConstDim::STIFFNESS)
        .with("g", ConstDim::ACCELERATION)
        .with("F", ConstDim::FORCE)
        .with("R", ConstDim::RESISTANCE)
        .with("I", ConstDim::CURRENT)
}

proptest! {
    #[test]
    fn simplify_preserves_force_dimension(coeff_a in 1..50i64, coeff_b in 1..50i64) {
        let ctx = Context::new();
        symplex::syms!(ctx; m, a);
        let dims = physics_dims();
        // Build a Force expression: (coeff_a*m*a + coeff_b*m*a) = (coeff_a+coeff_b)*m*a
        let f = Force::from_ex(
            &(ctx.int(coeff_a) * &m * &a) + &(ctx.int(coeff_b) * &m * &a),
        );
        let simplified = f.simplify();
        let dim = infer_dimension(simplified.inner(), &dims).unwrap();
        prop_assert!(
            dim.eq(ConstDim::FORCE),
            "simplify should preserve Force dimension, got {}", dim
        );
    }

    #[test]
    fn expand_preserves_energy_dimension(n in 2..6i64) {
        let ctx = Context::new();
        symplex::syms!(ctx; m, v);
        let dims = physics_dims();
        // Build Energy: n * m * v * v  (has dimension M·L²·T⁻² = Energy)
        let e = Energy::from_ex(ctx.int(n) * &m * &v * &v);
        let expanded = e.expand();
        let dim = infer_dimension(expanded.inner(), &dims).unwrap();
        prop_assert!(
            dim.eq(ConstDim::ENERGY),
            "expand should preserve Energy dimension, got {}", dim
        );
    }

    #[test]
    fn eval_preserves_force_dimension(c in 1..100i64) {
        let ctx = Context::new();
        symplex::syms!(ctx; m, a);
        let dims = physics_dims();
        let f = Force::from_ex(ctx.int(c) * &m * &a);
        let evaluated = f.eval();
        let dim = infer_dimension(evaluated.inner(), &dims).unwrap();
        prop_assert!(
            dim.eq(ConstDim::FORCE),
            "eval should preserve Force dimension, got {}", dim
        );
    }

    #[test]
    fn subs_preserves_dimension_when_value_matches(val in 1..50i64) {
        let ctx = Context::new();
        symplex::syms!(ctx; m, a, g);
        let dims = physics_dims();
        // F = m * a, substitute a → val * g (both Acceleration)
        let f = Force::from_ex(&m * &a);
        let f2 = f.subs(&a, &(ctx.int(val) * &g));
        let dim = infer_dimension(f2.inner(), &dims).unwrap();
        prop_assert!(
            dim.eq(ConstDim::FORCE),
            "subs with matching dimension should preserve Force, got {}", dim
        );
    }

    #[test]
    fn checked_from_ex_catches_mismatch(coeff in 1..50i64) {
        let ctx = Context::new();
        symplex::syms!(ctx; m, v);
        let dims = physics_dims();
        // m*v has dimension Momentum (L·M·T⁻¹), not Force (L·M·T⁻²)
        let momentum_expr = &m * &v * ctx.int(coeff);
        let result = Force::checked_from_ex(momentum_expr, &dims);
        prop_assert!(
            result.is_err(),
            "checked_from_ex should reject Momentum as Force"
        );
    }

    #[test]
    fn checked_from_ex_accepts_correct_dimension(coeff in 1..50i64) {
        let ctx = Context::new();
        symplex::syms!(ctx; m, a);
        let dims = physics_dims();
        // m*a has dimension Force — should be accepted
        let force_expr = ctx.int(coeff) * &m * &a;
        let result = Force::checked_from_ex(force_expr, &dims);
        prop_assert!(
            result.is_ok(),
            "checked_from_ex should accept valid Force expression, got {:?}", result.err()
        );
    }

    #[test]
    fn scalar_mul_preserves_dimension(coeff in 1..100i64) {
        let ctx = Context::new();
        symplex::syms!(ctx; m, a);
        let dims = physics_dims();
        let f = Force::from_ex(&m * &a);
        let scaled = f * coeff;
        let dim = infer_dimension(scaled.inner(), &dims).unwrap();
        prop_assert!(
            dim.eq(ConstDim::FORCE),
            "scalar multiplication should preserve Force dimension, got {}", dim
        );
    }

    #[test]
    fn add_preserves_dimension(coeff_a in 1..50i64, coeff_b in 1..50i64) {
        let ctx = Context::new();
        symplex::syms!(ctx; m, a);
        let dims = physics_dims();
        let f1 = Force::from_ex(ctx.int(coeff_a) * &m * &a);
        let f2 = Force::from_ex(ctx.int(coeff_b) * &m * &a);
        let sum = &f1 + &f2;
        let dim = infer_dimension(sum.inner(), &dims).unwrap();
        prop_assert!(
            dim.eq(ConstDim::FORCE),
            "addition of Forces should preserve Force dimension, got {}", dim
        );
    }

    #[test]
    fn simplify_preserves_velocity_dimension(a in 1..50i64, b in 1..50i64) {
        let ctx = Context::new();
        symplex::syms!(ctx; v, t);
        let dims = physics_dims();
        let vel = Velocity::from_ex(&(ctx.int(a) * &v) + &(ctx.int(b) * &v));
        let simplified = vel.simplify();
        let dim = infer_dimension(simplified.inner(), &dims).unwrap();
        prop_assert!(dim.eq(ConstDim::VELOCITY),
            "simplify should preserve Velocity dimension, got {}", dim);
    }

    #[allow(non_snake_case)]
    #[test]
    fn simplify_preserves_voltage_dimension(a in 1..50i64, b in 1..50i64) {
        let ctx = Context::new();
        symplex::syms!(ctx; I, R);
        let dims = physics_dims();
        let v = Voltage::from_ex(&(ctx.int(a) * &I * &R) + &(ctx.int(b) * &I * &R));
        let simplified = v.simplify();
        let dim = infer_dimension(simplified.inner(), &dims).unwrap();
        prop_assert!(dim.eq(ConstDim::VOLTAGE),
            "simplify should preserve Voltage dimension, got {}", dim);
    }

    #[test]
    fn simplify_preserves_power_dimension(coeff in 1..50i64) {
        let ctx = Context::new();
        symplex::syms!(ctx; m, a, v);
        let dims = physics_dims();
        let p = Power::from_ex(ctx.int(coeff) * &m * &a * &v);
        let simplified = p.simplify();
        let dim = infer_dimension(simplified.inner(), &dims).unwrap();
        prop_assert!(dim.eq(ConstDim::POWER),
            "simplify should preserve Power dimension, got {}", dim);
    }

    #[test]
    fn simplify_preserves_momentum_dimension(a in 1..50i64) {
        let ctx = Context::new();
        symplex::syms!(ctx; m, v);
        let dims = physics_dims();
        let p = Momentum::from_ex(ctx.int(a) * &m * &v);
        let simplified = p.simplify();
        let dim = infer_dimension(simplified.inner(), &dims).unwrap();
        prop_assert!(dim.eq(ConstDim::MOMENTUM),
            "simplify should preserve Momentum dimension, got {}", dim);
    }
}
