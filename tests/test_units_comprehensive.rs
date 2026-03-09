//! Comprehensive integration tests for the `symplex::units` module.
//!
//! Covers: DiffWrt, IntWrt, FTC round-trips, dim! macro numerical checks,
//! `from_ex` with IntoEx, and dimension-preserving operations.

use symplex::prelude::*;
use symplex::units::*;

// ═══════════════════════════════════════════════════════════════════════════
// Section 1: DiffWrt tests (~17 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_length_wrt_time_is_velocity() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; a, t);
    let x = Length::from_ex(expr!(a * t));
    let t_var = Time::symbol("t");
    let v: Velocity = x.diff_wrt(&t_var);
    assert_eq!(format!("{}", v.inner()), "a");
}

#[test]
fn diff_velocity_wrt_time_is_acceleration() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; a, t);
    let v = Velocity::from_ex(expr!(a * t));
    let t_var = Time::symbol("t");
    let acc: Acceleration = v.diff_wrt(&t_var);
    assert_eq!(format!("{}", acc.inner()), "a");
}

#[test]
fn diff_angle_wrt_time_is_angular_velocity() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; w, t);
    let theta = Angle::from_ex(expr!(w * t));
    let t_var = Time::symbol("t");
    let omega: AngularVelocity = theta.diff_wrt(&t_var);
    assert_eq!(format!("{}", omega.inner()), "w");
}

#[test]
fn diff_angular_velocity_wrt_time_is_angular_acceleration() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; a, t);
    let omega = AngularVelocity::from_ex(expr!(a * t));
    let t_var = Time::symbol("t");
    let alpha: AngularAcceleration = omega.diff_wrt(&t_var);
    assert_eq!(format!("{}", alpha.inner()), "a");
}

#[test]
fn diff_energy_wrt_time_is_power() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; p, t);
    let e = Energy::from_ex(expr!(p * t));
    let t_var = Time::symbol("t");
    let pwr: Power = e.diff_wrt(&t_var);
    assert_eq!(format!("{}", pwr.inner()), "p");
}

#[test]
fn diff_momentum_wrt_time_is_force() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; f, t);
    let p = Momentum::from_ex(expr!(f * t));
    let t_var = Time::symbol("t");
    let force: Force = p.diff_wrt(&t_var);
    assert_eq!(format!("{}", force.inner()), "f");
}

#[test]
fn diff_angular_momentum_wrt_time_is_torque() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; tau, t);
    let l = AngularMomentum::from_ex(expr!(tau * t));
    let t_var = Time::symbol("t");
    let torque: Torque = l.diff_wrt(&t_var);
    assert_eq!(format!("{}", torque.inner()), "tau");
}

#[test]
fn diff_charge_wrt_time_is_current() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; i, t);
    let q = Charge::from_ex(expr!(i * t));
    let t_var = Time::symbol("t");
    let current: Current = q.diff_wrt(&t_var);
    assert_eq!(format!("{}", current.inner()), "i");
}

#[test]
fn diff_magnetic_flux_wrt_time_is_voltage() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; v, t);
    let phi = MagneticFlux::from_ex(expr!(v * t));
    let t_var = Time::symbol("t");
    let voltage: Voltage = phi.diff_wrt(&t_var);
    assert_eq!(format!("{}", voltage.inner()), "v");
}

#[test]
fn diff_energy_wrt_length_is_force() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; f, x);
    let e = Energy::from_ex(expr!(f * x));
    let x_var = Length::symbol("x");
    let force: Force = e.diff_wrt(&x_var);
    assert_eq!(format!("{}", force.inner()), "f");
}

#[test]
fn diff_energy_wrt_angle_is_torque() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; tau, theta);
    let e = Energy::from_ex(expr!(tau * theta));
    let th_var = Angle::symbol("theta");
    let torque: Torque = e.diff_wrt(&th_var);
    assert_eq!(format!("{}", torque.inner()), "tau");
}

#[test]
fn diff_energy_wrt_velocity_is_momentum() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; p, v);
    let e = Energy::from_ex(expr!(p * v));
    let v_var = Velocity::symbol("v");
    let mom: Momentum = e.diff_wrt(&v_var);
    assert_eq!(format!("{}", mom.inner()), "p");
}

#[test]
fn diff_energy_wrt_angular_velocity_is_angular_momentum() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; l, w);
    let e = Energy::from_ex(expr!(l * w));
    let w_var = AngularVelocity::symbol("w");
    let am: AngularMomentum = e.diff_wrt(&w_var);
    assert_eq!(format!("{}", am.inner()), "l");
}

#[test]
fn diff_power_wrt_current_is_voltage() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; v, i);
    let p = Power::from_ex(expr!(v * i));
    let i_var = Current::symbol("i");
    let voltage: Voltage = p.diff_wrt(&i_var);
    assert_eq!(format!("{}", voltage.inner()), "v");
}

#[test]
fn diff_power_wrt_voltage_is_current() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; i, v);
    let p = Power::from_ex(expr!(i * v));
    let v_var = Voltage::symbol("v");
    let current: Current = p.diff_wrt(&v_var);
    assert_eq!(format!("{}", current.inner()), "i");
}

#[test]
fn diff_momentum_wrt_length_is_stiffness() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; k, x);
    let p = Momentum::from_ex(expr!(k * x));
    let x_var = Length::symbol("x");
    let stiffness: Stiffness = p.diff_wrt(&x_var);
    assert_eq!(format!("{}", stiffness.inner()), "k");
}

#[test]
fn diff_force_wrt_length_is_stiffness() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; k, x);
    let f = Force::from_ex(expr!(k * x));
    let x_var = Length::symbol("x");
    let stiffness: Stiffness = f.diff_wrt(&x_var);
    assert_eq!(format!("{}", stiffness.inner()), "k");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 2: IntWrt tests (~13 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_velocity_wrt_time_is_length() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; v, t);
    let vel = Velocity::from_ex(expr!(v));
    let t_var = Time::symbol("t");
    let x: Length = vel.integrate_wrt(&t_var);
    assert_eq!(format!("{}", x.inner()), "t*v");
}

#[test]
fn int_acceleration_wrt_time_is_velocity() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; a, t);
    let acc = Acceleration::from_ex(expr!(a));
    let t_var = Time::symbol("t");
    let v: Velocity = acc.integrate_wrt(&t_var);
    assert_eq!(format!("{}", v.inner()), "a*t");
}

#[test]
fn int_angular_velocity_wrt_time_is_angle() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; w, t);
    let omega = AngularVelocity::from_ex(expr!(w));
    let t_var = Time::symbol("t");
    let theta: Angle = omega.integrate_wrt(&t_var);
    assert_eq!(format!("{}", theta.inner()), "t*w");
}

#[test]
fn int_angular_acceleration_wrt_time_is_angular_velocity() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; a, t);
    let alpha = AngularAcceleration::from_ex(expr!(a));
    let t_var = Time::symbol("t");
    let omega: AngularVelocity = alpha.integrate_wrt(&t_var);
    assert_eq!(format!("{}", omega.inner()), "a*t");
}

#[test]
fn int_power_wrt_time_is_energy() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; p, t);
    let pwr = Power::from_ex(expr!(p));
    let t_var = Time::symbol("t");
    let e: Energy = pwr.integrate_wrt(&t_var);
    assert_eq!(format!("{}", e.inner()), "p*t");
}

#[test]
fn int_force_wrt_time_is_momentum() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; f, t);
    let force = Force::from_ex(expr!(f));
    let t_var = Time::symbol("t");
    let p: Momentum = force.integrate_wrt(&t_var);
    assert_eq!(format!("{}", p.inner()), "f*t");
}

#[test]
fn int_torque_wrt_time_is_angular_momentum() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; tau, t);
    let torque = Torque::from_ex(expr!(tau));
    let t_var = Time::symbol("t");
    let l: AngularMomentum = torque.integrate_wrt(&t_var);
    assert_eq!(format!("{}", l.inner()), "t*tau");
}

#[test]
fn int_current_wrt_time_is_charge() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; i, t);
    let cur = Current::from_ex(expr!(i));
    let t_var = Time::symbol("t");
    let q: Charge = cur.integrate_wrt(&t_var);
    assert_eq!(format!("{}", q.inner()), "i*t");
}

#[test]
fn int_voltage_wrt_time_is_magnetic_flux() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; v, t);
    let volt = Voltage::from_ex(expr!(v));
    let t_var = Time::symbol("t");
    let phi: MagneticFlux = volt.integrate_wrt(&t_var);
    assert_eq!(format!("{}", phi.inner()), "t*v");
}

#[test]
fn int_force_wrt_length_is_energy() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; f, x);
    let force = Force::from_ex(expr!(f));
    let x_var = Length::symbol("x");
    let e: Energy = force.integrate_wrt(&x_var);
    assert_eq!(format!("{}", e.inner()), "f*x");
}

#[test]
fn int_stiffness_wrt_length_is_force() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; k, x);
    let stiff = Stiffness::from_ex(expr!(k));
    let x_var = Length::symbol("x");
    let f: Force = stiff.integrate_wrt(&x_var);
    assert_eq!(format!("{}", f.inner()), "k*x");
}

#[test]
fn int_momentum_wrt_velocity_is_energy() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; p, v);
    let mom = Momentum::from_ex(expr!(p));
    let v_var = Velocity::symbol("v");
    let e: Energy = mom.integrate_wrt(&v_var);
    assert_eq!(format!("{}", e.inner()), "p*v");
}

#[test]
fn int_angular_momentum_wrt_angular_velocity_is_energy() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; l, w);
    let am = AngularMomentum::from_ex(expr!(l));
    let w_var = AngularVelocity::symbol("w");
    let e: Energy = am.integrate_wrt(&w_var);
    assert_eq!(format!("{}", e.inner()), "l*w");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 3: FTC Round-trips (~5 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ftc_acceleration_through_velocity() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; a, t);
    let accel = Acceleration::from_ex(expr!(a));
    let t_var = Time::symbol("t");
    let vel: Velocity = accel.integrate_wrt(&t_var);
    let accel_back: Acceleration = vel.diff_wrt(&t_var);
    assert_eq!(format!("{}", accel_back.inner()), "a");
}

#[test]
fn ftc_force_through_momentum() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; f, t);
    let force = Force::from_ex(expr!(f));
    let t_var = Time::symbol("t");
    let mom: Momentum = force.integrate_wrt(&t_var);
    let force_back: Force = mom.diff_wrt(&t_var);
    assert_eq!(format!("{}", force_back.inner()), "f");
}

#[test]
fn ftc_power_through_energy() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; p, t);
    let pwr = Power::from_ex(expr!(p));
    let t_var = Time::symbol("t");
    let energy: Energy = pwr.integrate_wrt(&t_var);
    let pwr_back: Power = energy.diff_wrt(&t_var);
    assert_eq!(format!("{}", pwr_back.inner()), "p");
}

#[test]
fn ftc_current_through_charge() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; i, t);
    let cur = Current::from_ex(expr!(i));
    let t_var = Time::symbol("t");
    let charge: Charge = cur.integrate_wrt(&t_var);
    let cur_back: Current = charge.diff_wrt(&t_var);
    assert_eq!(format!("{}", cur_back.inner()), "i");
}

#[test]
fn ftc_force_through_energy_spatial() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; f, x);
    let force = Force::from_ex(expr!(f));
    let x_var = Length::symbol("x");
    let energy: Energy = force.integrate_wrt(&x_var);
    let force_back: Force = energy.diff_wrt(&x_var);
    assert_eq!(format!("{}", force_back.inner()), "f");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 4: Mul/Div numerical spot-checks (~15 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mul_mass_acceleration_equals_force() {
    let ctx = symplex::units::si::units_ctx().clone();
    let m = Mass::constant(10);
    let a = Acceleration::rational(981, 100);
    let f = symplex::dim!(Force: m * a);
    let val = f.eval_f64().unwrap();
    assert!((val - 98.1).abs() < 1e-10, "Expected 98.1, got {val}");
}

#[test]
fn mul_current_resistance_equals_voltage() {
    let ctx = symplex::units::si::units_ctx().clone();
    let i = Current::constant(2);
    let r = Resistance::constant(100);
    let v = symplex::dim!(Voltage: i * r);
    let val = v.eval_f64().unwrap();
    assert!((val - 200.0).abs() < 1e-10, "Expected 200, got {val}");
}

#[test]
fn mul_voltage_current_equals_power() {
    let ctx = symplex::units::si::units_ctx().clone();
    let v = Voltage::constant(200);
    let i = Current::constant(2);
    let p = symplex::dim!(Power: v * i);
    let val = p.eval_f64().unwrap();
    assert!((val - 400.0).abs() < 1e-10, "Expected 400, got {val}");
}

#[test]
fn mul_force_length_equals_energy() {
    let ctx = symplex::units::si::units_ctx().clone();
    let f = Force::constant(50);
    let x = Length::constant(3);
    let e = symplex::dim!(Energy: f * x);
    let val = e.eval_f64().unwrap();
    assert!((val - 150.0).abs() < 1e-10, "Expected 150, got {val}");
}

#[test]
fn mul_mass_velocity_equals_momentum() {
    let ctx = symplex::units::si::units_ctx().clone();
    let m = Mass::constant(5);
    let v = Velocity::constant(10);
    let p = symplex::dim!(Momentum: m * v);
    let val = p.eval_f64().unwrap();
    assert!((val - 50.0).abs() < 1e-10, "Expected 50, got {val}");
}

#[test]
fn div_length_time_equals_velocity() {
    let ctx = symplex::units::si::units_ctx().clone();
    let x = Length::constant(100);
    let t = Time::constant(10);
    let v = symplex::dim!(Velocity: x / t);
    let val = v.eval_f64().unwrap();
    assert!((val - 10.0).abs() < 1e-10, "Expected 10, got {val}");
}

#[test]
fn div_energy_time_equals_power() {
    let ctx = symplex::units::si::units_ctx().clone();
    let e = Energy::constant(1000);
    let t = Time::constant(10);
    let p = symplex::dim!(Power: e / t);
    let val = p.eval_f64().unwrap();
    assert!((val - 100.0).abs() < 1e-10, "Expected 100, got {val}");
}

#[test]
fn div_force_mass_equals_acceleration() {
    let ctx = symplex::units::si::units_ctx().clone();
    let f = Force::rational(981, 10);
    let m = Mass::constant(10);
    let a = symplex::dim!(Acceleration: f / m);
    let val = a.eval_f64().unwrap();
    assert!((val - 9.81).abs() < 1e-10, "Expected 9.81, got {val}");
}

#[test]
fn div_voltage_current_equals_resistance() {
    let ctx = symplex::units::si::units_ctx().clone();
    let v = Voltage::constant(200);
    let i = Current::constant(2);
    let r = symplex::dim!(Resistance: v / i);
    let val = r.eval_f64().unwrap();
    assert!((val - 100.0).abs() < 1e-10, "Expected 100, got {val}");
}

#[test]
fn div_power_voltage_equals_current() {
    let ctx = symplex::units::si::units_ctx().clone();
    let p = Power::constant(400);
    let v = Voltage::constant(200);
    let i = symplex::dim!(Current: p / v);
    let val = i.eval_f64().unwrap();
    assert!((val - 2.0).abs() < 1e-10, "Expected 2, got {val}");
}

#[test]
fn mul_stiffness_length_equals_force() {
    let ctx = symplex::units::si::units_ctx().clone();
    let k = Stiffness::constant(100);
    let x = Length::rational(1, 2);
    let f = symplex::dim!(Force: k * x);
    let val = f.eval_f64().unwrap();
    assert!((val - 50.0).abs() < 1e-10, "Expected 50, got {val}");
}

#[test]
fn mul_damping_velocity_equals_force() {
    let ctx = symplex::units::si::units_ctx().clone();
    let c = Damping::constant(10);
    let v = Velocity::constant(3);
    let f = symplex::dim!(Force: c * v);
    let val = f.eval_f64().unwrap();
    assert!((val - 30.0).abs() < 1e-10, "Expected 30, got {val}");
}

#[test]
fn mul_acceleration_time_equals_velocity() {
    let ctx = symplex::units::si::units_ctx().clone();
    let a = Acceleration::rational(981, 100);
    let t = Time::constant(2);
    let v = symplex::dim!(Velocity: a * t);
    let val = v.eval_f64().unwrap();
    assert!((val - 19.62).abs() < 1e-10, "Expected 19.62, got {val}");
}

#[test]
fn mul_power_time_equals_energy() {
    let ctx = symplex::units::si::units_ctx().clone();
    let p = Power::constant(100);
    let t = Time::constant(10);
    let e = symplex::dim!(Energy: p * t);
    let val = e.eval_f64().unwrap();
    assert!((val - 1000.0).abs() < 1e-10, "Expected 1000, got {val}");
}

#[test]
fn mul_current_time_equals_charge() {
    let ctx = symplex::units::si::units_ctx().clone();
    let i = Current::constant(5);
    let t = Time::constant(10);
    let q = symplex::dim!(Charge: i * t);
    let val = q.eval_f64().unwrap();
    assert!((val - 50.0).abs() < 1e-10, "Expected 50, got {val}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 5: from_ex with &Ex (IntoEx) tests (~3 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn from_ex_accepts_ref_ex() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x);
    // expr!(x) returns &Ex — from_ex should accept it without .clone()
    let l = Length::from_ex(expr!(x));
    assert_eq!(format!("{}", l.inner()), "x");
}

#[test]
fn from_ex_accepts_owned_ex() {
    let ctx = symplex::units::si::units_ctx().clone();
    let ex = ctx.symbol("x");
    let l = Length::from_ex(ex);
    assert_eq!(format!("{}", l.inner()), "x");
}

#[test]
fn from_ex_accepts_expr_compound() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; a, t);
    // expr!(a * t) produces an Ex from &a * &t
    let v = Velocity::from_ex(expr!(a * t));
    assert_eq!(format!("{}", v.inner()), "a*t");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 6: Dimension-preserving operations (~5 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_preserves_dimension() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x);
    let f = Force::from_ex(expr!(x + x));
    let f2 = f.simplify();
    // simplify should still produce a Force
    assert_eq!(format!("{}", f2.inner()), "2*x");
}

#[test]
fn expand_preserves_dimension() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; a, b);
    // (a + b)^2 expanded = a^2 + 2*a*b + b^2
    let e = Energy::from_ex(expr!((a + b) * (a + b)));
    let e2 = e.expand();
    // The result should still be Energy; check it contains expected terms
    let inner_str = format!("{}", e2.inner());
    assert!(
        inner_str.contains("a") && inner_str.contains("b"),
        "Expanded expression should contain a and b: {inner_str}"
    );
}

#[test]
fn eval_preserves_dimension() {
    let v = Velocity::rational(22, 7);
    let v2 = v.eval();
    // eval on a rational should keep it as-is (already evaluated)
    let val = v2.eval_f64().unwrap();
    assert!((val - 22.0 / 7.0).abs() < 1e-10);
}

#[test]
fn subs_preserves_dimension() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x, y);
    let f = Force::from_ex(expr!(x + y));
    let two = ctx.int(2);
    let f2 = f.subs(&x, &two);
    // After substituting x=2, should get 2 + y
    let inner_str = format!("{}", f2.inner());
    assert!(
        inner_str.contains("2") && inner_str.contains("y"),
        "After subs x=2, should contain 2 and y: {inner_str}"
    );
}

#[test]
fn to_latex_preserves_dimension() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; f);
    let force = Force::from_ex(expr!(f));
    let latex = force.to_latex();
    // to_latex should return a non-empty LaTeX string
    assert!(!latex.is_empty(), "LaTeX output should not be empty");
    assert!(latex.contains("f"), "LaTeX should contain the variable name f: {latex}");
}

// ═══════════════════════════════════════════════════════════════════════════
// AsRef<Ex> tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn asref_force_returns_ex() {
    let f = Force::constant(98);
    let ex: &Ex = f.as_ref();
    assert_eq!(format!("{}", ex), "98");
}

#[test]
fn asref_qty_returns_ex() {
    let ctx = symplex::units::si::units_ctx().clone();
    let q: Qty<LengthDim> = Qty::from_ex(ctx.int(42));
    let ex: &Ex = q.as_ref();
    assert_eq!(format!("{}", ex), "42");
}

#[test]
fn subs_accepts_named_type_var() {
    let ctx = symplex::units::si::units_ctx().clone();
    let m = Mass::symbol("m");
    let a = Acceleration::symbol("a");
    let f = symplex::dim!(Force: m * a);
    // subs with named type — no .inner() needed!
    let f2 = f.subs(&m, &ctx.int(10));
    assert_eq!(format!("{}", f2.inner()), "10*a");
}

#[test]
fn subs_still_accepts_raw_ex() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; m, a);
    let m_ex = ctx.symbol("m");
    let f = Force::from_ex(expr!(m * a));
    // subs with raw &Ex — backward compatible
    let f2 = f.subs(&m_ex, &ctx.int(10));
    assert!(format!("{}", f2.inner()).contains("10"));
}

#[test]
fn subs_chain_no_inner() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; m, a);
    let f = Force::from_ex(expr!(m * a));
    let m_var = Mass::symbol("m");
    let a_var = Acceleration::symbol("a");
    let result = f.subs(&m_var, &ctx.int(5)).subs(&a_var, &ctx.int(10)).eval();
    assert_eq!(format!("{}", result.inner()), "50");
}

#[test]
fn diff_accepts_named_type_var() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x, t);
    let pos = Length::from_ex(expr!(x * t));
    let t_var = Time::symbol("t");
    // diff with named type
    let result = pos.diff(&t_var);
    assert!(format!("{}", result).contains("x"));
}

#[test]
fn integrate_accepts_named_type_var() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; v, t);
    let vel = Velocity::from_ex(expr!(v));
    let t_var = Time::symbol("t");
    let result = vel.integrate(&t_var);
    assert!(format!("{}", result).contains("t"));
}

#[test]
fn contains_accepts_named_type() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; m, a);
    let f = Force::from_ex(expr!(m * a));
    let m_var = Mass::symbol("m");
    assert!(f.contains(&m_var));
}

#[test]
fn qty_subs_accepts_named_type() {
    let ctx = symplex::units::si::units_ctx().clone();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; m, a);
    let q: Qty<ForceDim> = Qty::from_ex(expr!(m * a));
    let m_var = Mass::symbol("m");
    let q2 = q.subs(&m_var, &ctx.int(7));
    assert!(format!("{}", q2).contains("7"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Qty Display without DimName
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn qty_display_known_dim() {
    let ctx = symplex::units::si::units_ctx().clone();
    let q: Qty<LengthDim> = Qty::from_ex(ctx.int(42));
    let s = format!("{}", q);
    assert_eq!(s, "42");
}

#[test]
fn qty_display_exotic_dim() {
    // Boltzmann dimension: Dim<P2, P1, N2, Z0, N1, Z0, Z0>
    // This used to fail because no DimName impl exists for this dimension
    use symplex::units::constants;
    let kb = constants::boltzmann_constant();
    let s = format!("{}", kb);
    assert!(s.contains("k_B"), "should display the constant name: {s}");
}

#[test]
fn qty_debug_exotic_dim() {
    use symplex::units::constants;
    let kb = constants::boltzmann_constant();
    let s = format!("{:?}", kb);
    assert!(s.contains("k_B"), "debug should contain constant name: {s}");
}

#[test]
fn named_display_keeps_suffix() {
    let f = Force::constant(98);
    let s = format!("{}", f);
    assert!(s.contains("[N]"), "Force display should keep unit suffix: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// New Mul chain entries
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn momentum_times_velocity_is_energy() {
    let ctx = symplex::units::si::units_ctx().clone();
    let p = Momentum::constant(10);
    let v = Velocity::constant(3);
    let e = symplex::dim!(Energy: p * v);
    assert_eq!(e.eval_f64().unwrap(), 30.0);
}

#[test]
fn velocity_times_momentum_is_energy() {
    let ctx = symplex::units::si::units_ctx().clone();
    let v = Velocity::constant(3);
    let p = Momentum::constant(10);
    let e = symplex::dim!(Energy: v * p);
    assert_eq!(e.eval_f64().unwrap(), 30.0);
}

#[test]
fn e_mc_squared_typed_chain() {
    let ctx = symplex::units::si::units_ctx().clone();
    let m = Mass::constant(1);
    use symplex::units::constants;
    let c = constants::speed_of_light();
    // Mass × Velocity × Velocity = Energy (via dim! macro)
    let e = symplex::dim!(Energy: m * c * c);
    let val = e.eval_f64().unwrap();
    let expected = 299792458.0_f64 * 299792458.0;
    assert!((val - expected).abs() / expected < 1e-10);
}

#[test]
fn angular_momentum_times_angvel_is_energy() {
    let ctx = symplex::units::si::units_ctx().clone();
    let l = AngularMomentum::constant(5);
    let w = AngularVelocity::constant(4);
    let e = symplex::dim!(Energy: l * w);
    assert_eq!(e.eval_f64().unwrap(), 20.0);
}

#[test]
fn half_i_omega_squared_chain() {
    let ctx = symplex::units::si::units_ctx().clone();
    // T = ½Iω²: MoI×AngVel×AngVel=Energy (via dim! macro)
    let i_moi = MomentOfInertia::constant(2);
    let w = AngularVelocity::constant(3);
    let ke = symplex::dim!(Energy: i_moi * w * w);
    // Result should be 2*3*3 = 18 (without the ½)
    assert_eq!(ke.eval_f64().unwrap(), 18.0);
}

#[test]
fn force_times_time_is_momentum() {
    let ctx = symplex::units::si::units_ctx().clone();
    let f = Force::constant(50);
    let t = Time::constant(2);
    let j = symplex::dim!(Momentum: f * t);
    assert_eq!(j.eval_f64().unwrap(), 100.0);
}

#[test]
fn capacitance_times_voltage_is_charge() {
    let ctx = symplex::units::si::units_ctx().clone();
    let cap = Capacitance::rational(1, 1000); // 1 mF
    let v = Voltage::constant(5);
    let q = symplex::dim!(Charge: cap * v);
    let val = q.eval_f64().unwrap();
    assert!((val - 0.005).abs() < 1e-10);
}

#[test]
fn pressure_times_area_is_force() {
    let ctx = symplex::units::si::units_ctx().clone();
    let p = Pressure::constant(100000); // 100 kPa
    let a = Area::constant(2);
    let f = symplex::dim!(Force: p * a);
    assert_eq!(f.eval_f64().unwrap(), 200000.0);
}

#[test]
fn pressure_times_volume_is_energy() {
    let ctx = symplex::units::si::units_ctx().clone();
    let p = Pressure::constant(101325); // 1 atm
    let v = Volume::constant(1);
    let e = symplex::dim!(Energy: p * v);
    assert_eq!(e.eval_f64().unwrap(), 101325.0);
}
