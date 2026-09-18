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
    let ctx = Context::new();
    symplex::syms!(ctx; a, t);
    let x = Length::from_ex(expr!(ctx, a * t));
    let t_var = Time::symbol(&ctx, "t");
    let v: Velocity = x.diff_wrt(&t_var);
    assert_eq!(format!("{}", v.inner()), "a");
}

#[test]
fn diff_velocity_wrt_time_is_acceleration() {
    let ctx = Context::new();
    symplex::syms!(ctx; a, t);
    let v = Velocity::from_ex(expr!(ctx, a * t));
    let t_var = Time::symbol(&ctx, "t");
    let acc: Acceleration = v.diff_wrt(&t_var);
    assert_eq!(format!("{}", acc.inner()), "a");
}

#[test]
fn diff_angle_wrt_time_is_angular_velocity() {
    let ctx = Context::new();
    symplex::syms!(ctx; w, t);
    let theta = Angle::from_ex(expr!(ctx, w * t));
    let t_var = Time::symbol(&ctx, "t");
    let omega: AngularVelocity = theta.diff_wrt(&t_var);
    assert_eq!(format!("{}", omega.inner()), "w");
}

#[test]
fn diff_angular_velocity_wrt_time_is_angular_acceleration() {
    let ctx = Context::new();
    symplex::syms!(ctx; a, t);
    let omega = AngularVelocity::from_ex(expr!(ctx, a * t));
    let t_var = Time::symbol(&ctx, "t");
    let alpha: AngularAcceleration = omega.diff_wrt(&t_var);
    assert_eq!(format!("{}", alpha.inner()), "a");
}

#[test]
fn diff_energy_wrt_time_is_power() {
    let ctx = Context::new();
    symplex::syms!(ctx; p, t);
    let e = Energy::from_ex(expr!(ctx, p * t));
    let t_var = Time::symbol(&ctx, "t");
    let pwr: Power = e.diff_wrt(&t_var);
    assert_eq!(format!("{}", pwr.inner()), "p");
}

#[test]
fn diff_momentum_wrt_time_is_force() {
    let ctx = Context::new();
    symplex::syms!(ctx; f, t);
    let p = Momentum::from_ex(expr!(ctx, f * t));
    let t_var = Time::symbol(&ctx, "t");
    let force: Force = p.diff_wrt(&t_var);
    assert_eq!(format!("{}", force.inner()), "f");
}

#[test]
fn diff_angular_momentum_wrt_time_is_torque() {
    let ctx = Context::new();
    symplex::syms!(ctx; tau, t);
    let l = AngularMomentum::from_ex(expr!(ctx, tau * t));
    let t_var = Time::symbol(&ctx, "t");
    let torque: Torque = l.diff_wrt(&t_var);
    assert_eq!(format!("{}", torque.inner()), "tau");
}

#[test]
fn diff_charge_wrt_time_is_current() {
    let ctx = Context::new();
    symplex::syms!(ctx; i, t);
    let q = Charge::from_ex(expr!(ctx, i * t));
    let t_var = Time::symbol(&ctx, "t");
    let current: Current = q.diff_wrt(&t_var);
    assert_eq!(format!("{}", current.inner()), "i");
}

#[test]
fn diff_magnetic_flux_wrt_time_is_voltage() {
    let ctx = Context::new();
    symplex::syms!(ctx; v, t);
    let phi = MagneticFlux::from_ex(expr!(ctx, v * t));
    let t_var = Time::symbol(&ctx, "t");
    let voltage: Voltage = phi.diff_wrt(&t_var);
    assert_eq!(format!("{}", voltage.inner()), "v");
}

#[test]
fn diff_energy_wrt_length_is_force() {
    let ctx = Context::new();
    symplex::syms!(ctx; f, x);
    let e = Energy::from_ex(expr!(ctx, f * x));
    let x_var = Length::symbol(&ctx, "x");
    let force: Force = e.diff_wrt(&x_var);
    assert_eq!(format!("{}", force.inner()), "f");
}

#[test]
fn diff_energy_wrt_angle_is_torque() {
    let ctx = Context::new();
    symplex::syms!(ctx; tau, theta);
    let e = Energy::from_ex(expr!(ctx, tau * theta));
    let th_var = Angle::symbol(&ctx, "theta");
    let torque: Torque = e.diff_wrt(&th_var);
    assert_eq!(format!("{}", torque.inner()), "tau");
}

#[test]
fn diff_energy_wrt_velocity_is_momentum() {
    let ctx = Context::new();
    symplex::syms!(ctx; p, v);
    let e = Energy::from_ex(expr!(ctx, p * v));
    let v_var = Velocity::symbol(&ctx, "v");
    let mom: Momentum = e.diff_wrt(&v_var);
    assert_eq!(format!("{}", mom.inner()), "p");
}

#[test]
fn diff_energy_wrt_angular_velocity_is_angular_momentum() {
    let ctx = Context::new();
    symplex::syms!(ctx; l, w);
    let e = Energy::from_ex(expr!(ctx, l * w));
    let w_var = AngularVelocity::symbol(&ctx, "w");
    let am: AngularMomentum = e.diff_wrt(&w_var);
    assert_eq!(format!("{}", am.inner()), "l");
}

#[test]
fn diff_power_wrt_current_is_voltage() {
    let ctx = Context::new();
    symplex::syms!(ctx; v, i);
    let p = Power::from_ex(expr!(ctx, v * i));
    let i_var = Current::symbol(&ctx, "i");
    let voltage: Voltage = p.diff_wrt(&i_var);
    assert_eq!(format!("{}", voltage.inner()), "v");
}

#[test]
fn diff_power_wrt_voltage_is_current() {
    let ctx = Context::new();
    symplex::syms!(ctx; i, v);
    let p = Power::from_ex(expr!(ctx, i * v));
    let v_var = Voltage::symbol(&ctx, "v");
    let current: Current = p.diff_wrt(&v_var);
    assert_eq!(format!("{}", current.inner()), "i");
}

#[test]
fn diff_momentum_wrt_length_is_stiffness() {
    let ctx = Context::new();
    symplex::syms!(ctx; k, x);
    let p = Momentum::from_ex(expr!(ctx, k * x));
    let x_var = Length::symbol(&ctx, "x");
    let stiffness: Stiffness = p.diff_wrt(&x_var);
    assert_eq!(format!("{}", stiffness.inner()), "k");
}

#[test]
fn diff_force_wrt_length_is_stiffness() {
    let ctx = Context::new();
    symplex::syms!(ctx; k, x);
    let f = Force::from_ex(expr!(ctx, k * x));
    let x_var = Length::symbol(&ctx, "x");
    let stiffness: Stiffness = f.diff_wrt(&x_var);
    assert_eq!(format!("{}", stiffness.inner()), "k");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 2: IntWrt tests (~13 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_velocity_wrt_time_is_length() {
    let ctx = Context::new();
    symplex::syms!(ctx; v, t);
    let vel = Velocity::from_ex(expr!(ctx, v));
    let t_var = Time::symbol(&ctx, "t");
    let x: Length = vel.integrate_wrt(&t_var);
    assert_eq!(format!("{}", x.inner()), "t*v");
}

#[test]
fn int_acceleration_wrt_time_is_velocity() {
    let ctx = Context::new();
    symplex::syms!(ctx; a, t);
    let acc = Acceleration::from_ex(expr!(ctx, a));
    let t_var = Time::symbol(&ctx, "t");
    let v: Velocity = acc.integrate_wrt(&t_var);
    assert_eq!(format!("{}", v.inner()), "a*t");
}

#[test]
fn int_angular_velocity_wrt_time_is_angle() {
    let ctx = Context::new();
    symplex::syms!(ctx; w, t);
    let omega = AngularVelocity::from_ex(expr!(ctx, w));
    let t_var = Time::symbol(&ctx, "t");
    let theta: Angle = omega.integrate_wrt(&t_var);
    assert_eq!(format!("{}", theta.inner()), "t*w");
}

#[test]
fn int_angular_acceleration_wrt_time_is_angular_velocity() {
    let ctx = Context::new();
    symplex::syms!(ctx; a, t);
    let alpha = AngularAcceleration::from_ex(expr!(ctx, a));
    let t_var = Time::symbol(&ctx, "t");
    let omega: AngularVelocity = alpha.integrate_wrt(&t_var);
    assert_eq!(format!("{}", omega.inner()), "a*t");
}

#[test]
fn int_power_wrt_time_is_energy() {
    let ctx = Context::new();
    symplex::syms!(ctx; p, t);
    let pwr = Power::from_ex(expr!(ctx, p));
    let t_var = Time::symbol(&ctx, "t");
    let e: Energy = pwr.integrate_wrt(&t_var);
    assert_eq!(format!("{}", e.inner()), "p*t");
}

#[test]
fn int_force_wrt_time_is_momentum() {
    let ctx = Context::new();
    symplex::syms!(ctx; f, t);
    let force = Force::from_ex(expr!(ctx, f));
    let t_var = Time::symbol(&ctx, "t");
    let p: Momentum = force.integrate_wrt(&t_var);
    assert_eq!(format!("{}", p.inner()), "f*t");
}

#[test]
fn int_torque_wrt_time_is_angular_momentum() {
    let ctx = Context::new();
    symplex::syms!(ctx; tau, t);
    let torque = Torque::from_ex(expr!(ctx, tau));
    let t_var = Time::symbol(&ctx, "t");
    let l: AngularMomentum = torque.integrate_wrt(&t_var);
    assert_eq!(format!("{}", l.inner()), "t*tau");
}

#[test]
fn int_current_wrt_time_is_charge() {
    let ctx = Context::new();
    symplex::syms!(ctx; i, t);
    let cur = Current::from_ex(expr!(ctx, i));
    let t_var = Time::symbol(&ctx, "t");
    let q: Charge = cur.integrate_wrt(&t_var);
    assert_eq!(format!("{}", q.inner()), "i*t");
}

#[test]
fn int_voltage_wrt_time_is_magnetic_flux() {
    let ctx = Context::new();
    symplex::syms!(ctx; v, t);
    let volt = Voltage::from_ex(expr!(ctx, v));
    let t_var = Time::symbol(&ctx, "t");
    let phi: MagneticFlux = volt.integrate_wrt(&t_var);
    assert_eq!(format!("{}", phi.inner()), "t*v");
}

#[test]
fn int_force_wrt_length_is_energy() {
    let ctx = Context::new();
    symplex::syms!(ctx; f, x);
    let force = Force::from_ex(expr!(ctx, f));
    let x_var = Length::symbol(&ctx, "x");
    let e: Energy = force.integrate_wrt(&x_var);
    assert_eq!(format!("{}", e.inner()), "f*x");
}

#[test]
fn int_stiffness_wrt_length_is_force() {
    let ctx = Context::new();
    symplex::syms!(ctx; k, x);
    let stiff = Stiffness::from_ex(expr!(ctx, k));
    let x_var = Length::symbol(&ctx, "x");
    let f: Force = stiff.integrate_wrt(&x_var);
    assert_eq!(format!("{}", f.inner()), "k*x");
}

#[test]
fn int_momentum_wrt_velocity_is_energy() {
    let ctx = Context::new();
    symplex::syms!(ctx; p, v);
    let mom = Momentum::from_ex(expr!(ctx, p));
    let v_var = Velocity::symbol(&ctx, "v");
    let e: Energy = mom.integrate_wrt(&v_var);
    assert_eq!(format!("{}", e.inner()), "p*v");
}

#[test]
fn int_angular_momentum_wrt_angular_velocity_is_energy() {
    let ctx = Context::new();
    symplex::syms!(ctx; l, w);
    let am = AngularMomentum::from_ex(expr!(ctx, l));
    let w_var = AngularVelocity::symbol(&ctx, "w");
    let e: Energy = am.integrate_wrt(&w_var);
    assert_eq!(format!("{}", e.inner()), "l*w");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 3: FTC Round-trips (~5 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ftc_acceleration_through_velocity() {
    let ctx = Context::new();
    symplex::syms!(ctx; a, t);
    let accel = Acceleration::from_ex(expr!(ctx, a));
    let t_var = Time::symbol(&ctx, "t");
    let vel: Velocity = accel.integrate_wrt(&t_var);
    let accel_back: Acceleration = vel.diff_wrt(&t_var);
    assert_eq!(format!("{}", accel_back.inner()), "a");
}

#[test]
fn ftc_force_through_momentum() {
    let ctx = Context::new();
    symplex::syms!(ctx; f, t);
    let force = Force::from_ex(expr!(ctx, f));
    let t_var = Time::symbol(&ctx, "t");
    let mom: Momentum = force.integrate_wrt(&t_var);
    let force_back: Force = mom.diff_wrt(&t_var);
    assert_eq!(format!("{}", force_back.inner()), "f");
}

#[test]
fn ftc_power_through_energy() {
    let ctx = Context::new();
    symplex::syms!(ctx; p, t);
    let pwr = Power::from_ex(expr!(ctx, p));
    let t_var = Time::symbol(&ctx, "t");
    let energy: Energy = pwr.integrate_wrt(&t_var);
    let pwr_back: Power = energy.diff_wrt(&t_var);
    assert_eq!(format!("{}", pwr_back.inner()), "p");
}

#[test]
fn ftc_current_through_charge() {
    let ctx = Context::new();
    symplex::syms!(ctx; i, t);
    let cur = Current::from_ex(expr!(ctx, i));
    let t_var = Time::symbol(&ctx, "t");
    let charge: Charge = cur.integrate_wrt(&t_var);
    let cur_back: Current = charge.diff_wrt(&t_var);
    assert_eq!(format!("{}", cur_back.inner()), "i");
}

#[test]
fn ftc_force_through_energy_spatial() {
    let ctx = Context::new();
    symplex::syms!(ctx; f, x);
    let force = Force::from_ex(expr!(ctx, f));
    let x_var = Length::symbol(&ctx, "x");
    let energy: Energy = force.integrate_wrt(&x_var);
    let force_back: Force = energy.diff_wrt(&x_var);
    assert_eq!(format!("{}", force_back.inner()), "f");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 4: Mul/Div numerical spot-checks (~15 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mul_mass_acceleration_equals_force() {
    let ctx = Context::new();
    let m = Mass::constant(&ctx, 10);
    let a = Acceleration::rational(&ctx, 981, 100);
    let f = symplex::dim!(ctx, Force: m * a);
    let val = f.eval_f64().unwrap();
    assert!((val - 98.1).abs() < 1e-10, "Expected 98.1, got {val}");
}

#[test]
fn mul_current_resistance_equals_voltage() {
    let ctx = Context::new();
    let i = Current::constant(&ctx, 2);
    let r = Resistance::constant(&ctx, 100);
    let v = symplex::dim!(ctx, Voltage: i * r);
    let val = v.eval_f64().unwrap();
    assert!((val - 200.0).abs() < 1e-10, "Expected 200, got {val}");
}

#[test]
fn mul_voltage_current_equals_power() {
    let ctx = Context::new();
    let v = Voltage::constant(&ctx, 200);
    let i = Current::constant(&ctx, 2);
    let p = symplex::dim!(ctx, Power: v * i);
    let val = p.eval_f64().unwrap();
    assert!((val - 400.0).abs() < 1e-10, "Expected 400, got {val}");
}

#[test]
fn mul_force_length_equals_energy() {
    let ctx = Context::new();
    let f = Force::constant(&ctx, 50);
    let x = Length::constant(&ctx, 3);
    let e = symplex::dim!(ctx, Energy: f * x);
    let val = e.eval_f64().unwrap();
    assert!((val - 150.0).abs() < 1e-10, "Expected 150, got {val}");
}

#[test]
fn mul_mass_velocity_equals_momentum() {
    let ctx = Context::new();
    let m = Mass::constant(&ctx, 5);
    let v = Velocity::constant(&ctx, 10);
    let p = symplex::dim!(ctx, Momentum: m * v);
    let val = p.eval_f64().unwrap();
    assert!((val - 50.0).abs() < 1e-10, "Expected 50, got {val}");
}

#[test]
fn div_length_time_equals_velocity() {
    let ctx = Context::new();
    let x = Length::constant(&ctx, 100);
    let t = Time::constant(&ctx, 10);
    let v = symplex::dim!(ctx, Velocity: x / t);
    let val = v.eval_f64().unwrap();
    assert!((val - 10.0).abs() < 1e-10, "Expected 10, got {val}");
}

#[test]
fn div_energy_time_equals_power() {
    let ctx = Context::new();
    let e = Energy::constant(&ctx, 1000);
    let t = Time::constant(&ctx, 10);
    let p = symplex::dim!(ctx, Power: e / t);
    let val = p.eval_f64().unwrap();
    assert!((val - 100.0).abs() < 1e-10, "Expected 100, got {val}");
}

#[test]
fn div_force_mass_equals_acceleration() {
    let ctx = Context::new();
    let f = Force::rational(&ctx, 981, 10);
    let m = Mass::constant(&ctx, 10);
    let a = symplex::dim!(ctx, Acceleration: f / m);
    let val = a.eval_f64().unwrap();
    assert!((val - 9.81).abs() < 1e-10, "Expected 9.81, got {val}");
}

#[test]
fn div_voltage_current_equals_resistance() {
    let ctx = Context::new();
    let v = Voltage::constant(&ctx, 200);
    let i = Current::constant(&ctx, 2);
    let r = symplex::dim!(ctx, Resistance: v / i);
    let val = r.eval_f64().unwrap();
    assert!((val - 100.0).abs() < 1e-10, "Expected 100, got {val}");
}

#[test]
fn div_power_voltage_equals_current() {
    let ctx = Context::new();
    let p = Power::constant(&ctx, 400);
    let v = Voltage::constant(&ctx, 200);
    let i = symplex::dim!(ctx, Current: p / v);
    let val = i.eval_f64().unwrap();
    assert!((val - 2.0).abs() < 1e-10, "Expected 2, got {val}");
}

#[test]
fn mul_stiffness_length_equals_force() {
    let ctx = Context::new();
    let k = Stiffness::constant(&ctx, 100);
    let x = Length::rational(&ctx, 1, 2);
    let f = symplex::dim!(ctx, Force: k * x);
    let val = f.eval_f64().unwrap();
    assert!((val - 50.0).abs() < 1e-10, "Expected 50, got {val}");
}

#[test]
fn mul_damping_velocity_equals_force() {
    let ctx = Context::new();
    let c = Damping::constant(&ctx, 10);
    let v = Velocity::constant(&ctx, 3);
    let f = symplex::dim!(ctx, Force: c * v);
    let val = f.eval_f64().unwrap();
    assert!((val - 30.0).abs() < 1e-10, "Expected 30, got {val}");
}

#[test]
fn mul_acceleration_time_equals_velocity() {
    let ctx = Context::new();
    let a = Acceleration::rational(&ctx, 981, 100);
    let t = Time::constant(&ctx, 2);
    let v = symplex::dim!(ctx, Velocity: a * t);
    let val = v.eval_f64().unwrap();
    assert!((val - 19.62).abs() < 1e-10, "Expected 19.62, got {val}");
}

#[test]
fn mul_power_time_equals_energy() {
    let ctx = Context::new();
    let p = Power::constant(&ctx, 100);
    let t = Time::constant(&ctx, 10);
    let e = symplex::dim!(ctx, Energy: p * t);
    let val = e.eval_f64().unwrap();
    assert!((val - 1000.0).abs() < 1e-10, "Expected 1000, got {val}");
}

#[test]
fn mul_current_time_equals_charge() {
    let ctx = Context::new();
    let i = Current::constant(&ctx, 5);
    let t = Time::constant(&ctx, 10);
    let q = symplex::dim!(ctx, Charge: i * t);
    let val = q.eval_f64().unwrap();
    assert!((val - 50.0).abs() < 1e-10, "Expected 50, got {val}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 5: from_ex with &Ex (IntoEx) tests (~3 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn from_ex_accepts_ref_ex() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // expr!(ctx, x) returns &Ex — from_ex should accept it without .clone()
    let l = Length::from_ex(expr!(ctx, x));
    assert_eq!(format!("{}", l.inner()), "x");
}

#[test]
fn from_ex_accepts_owned_ex() {
    let ctx = Context::new();
    let ex = ctx.symbol("x");
    let l = Length::from_ex(ex);
    assert_eq!(format!("{}", l.inner()), "x");
}

#[test]
fn from_ex_accepts_expr_compound() {
    let ctx = Context::new();
    symplex::syms!(ctx; a, t);
    // expr!(ctx, a * t) produces an Ex from &a * &t
    let v = Velocity::from_ex(expr!(ctx, a * t));
    assert_eq!(format!("{}", v.inner()), "a*t");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 6: Dimension-preserving operations (~5 tests)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_preserves_dimension() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let f = Force::from_ex(expr!(ctx, x + x));
    let f2 = f.simplify();
    // simplify should still produce a Force
    assert_eq!(format!("{}", f2.inner()), "2*x");
}

#[test]
fn expand_preserves_dimension() {
    let ctx = Context::new();
    symplex::syms!(ctx; a, b);
    // (a + b)^2 expanded = a^2 + 2*a*b + b^2
    let e = Energy::from_ex(expr!(ctx, (a + b) * (a + b)));
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
    let ctx = Context::new();
    let v = Velocity::rational(&ctx, 22, 7);
    let v2 = v.eval();
    // eval on a rational should keep it as-is (already evaluated)
    let val = v2.eval_f64().unwrap();
    assert!((val - 22.0 / 7.0).abs() < 1e-10);
}

#[test]
fn subs_preserves_dimension() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let f = Force::from_ex(expr!(ctx, x + y));
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
    let ctx = Context::new();
    symplex::syms!(ctx; f);
    let force = Force::from_ex(expr!(ctx, f));
    let latex = force.to_latex();
    // to_latex should return a non-empty LaTeX string
    assert!(!latex.is_empty(), "LaTeX output should not be empty");
    assert!(
        latex.contains("f"),
        "LaTeX should contain the variable name f: {latex}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// AsRef<Ex> tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn asref_force_returns_ex() {
    let ctx = Context::new();
    let f = Force::constant(&ctx, 98);
    let ex: &Ex = f.as_ref();
    assert_eq!(format!("{}", ex), "98");
}

#[test]
fn asref_qty_returns_ex() {
    let ctx = Context::new();
    let q: Qty<LengthDim> = Qty::from_ex(ctx.int(42));
    let ex: &Ex = q.as_ref();
    assert_eq!(format!("{}", ex), "42");
}

#[test]
fn subs_accepts_named_type_var() {
    let ctx = Context::new();
    let m = Mass::symbol(&ctx, "m");
    let a = Acceleration::symbol(&ctx, "a");
    let f = symplex::dim!(ctx, Force: m * a);
    // subs with named type — no .inner() needed!
    let f2 = f.subs(&m, &ctx.int(10));
    assert_eq!(format!("{}", f2.inner()), "10*a");
}

#[test]
fn subs_still_accepts_raw_ex() {
    let ctx = Context::new();
    symplex::syms!(ctx; m, a);
    let m_ex = ctx.symbol("m");
    let f = Force::from_ex(expr!(ctx, m * a));
    // subs with raw &Ex — backward compatible
    let f2 = f.subs(&m_ex, &ctx.int(10));
    assert!(format!("{}", f2.inner()).contains("10"));
}

#[test]
fn subs_chain_no_inner() {
    let ctx = Context::new();
    symplex::syms!(ctx; m, a);
    let f = Force::from_ex(expr!(ctx, m * a));
    let m_var = Mass::symbol(&ctx, "m");
    let a_var = Acceleration::symbol(&ctx, "a");
    let result = f
        .subs(&m_var, &ctx.int(5))
        .subs(&a_var, &ctx.int(10))
        .eval();
    assert_eq!(format!("{}", result.inner()), "50");
}

#[test]
fn diff_accepts_named_type_var() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, t);
    let pos = Length::from_ex(expr!(ctx, x * t));
    let t_var = Time::symbol(&ctx, "t");
    // diff with named type
    let result = pos.diff(&t_var);
    assert!(format!("{}", result).contains("x"));
}

#[test]
fn integrate_accepts_named_type_var() {
    let ctx = Context::new();
    symplex::syms!(ctx; v, t);
    let vel = Velocity::from_ex(expr!(ctx, v));
    let t_var = Time::symbol(&ctx, "t");
    let result = vel.integrate(&t_var);
    assert!(format!("{}", result).contains("t"));
}

#[test]
fn contains_accepts_named_type() {
    let ctx = Context::new();
    symplex::syms!(ctx; m, a);
    let f = Force::from_ex(expr!(ctx, m * a));
    let m_var = Mass::symbol(&ctx, "m");
    assert!(f.contains(&m_var));
}

#[test]
fn qty_subs_accepts_named_type() {
    let ctx = Context::new();
    symplex::syms!(ctx; m, a);
    let q: Qty<ForceDim> = Qty::from_ex(expr!(ctx, m * a));
    let m_var = Mass::symbol(&ctx, "m");
    let q2 = q.subs(&m_var, &ctx.int(7));
    assert!(format!("{}", q2).contains("7"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Qty Display without DimName
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn qty_display_known_dim() {
    let ctx = Context::new();
    let q: Qty<LengthDim> = Qty::from_ex(ctx.int(42));
    let s = format!("{}", q);
    assert_eq!(s, "42");
}

#[test]
fn qty_display_exotic_dim() {
    // Boltzmann dimension: Dim<P2, P1, N2, Z0, N1, Z0, Z0>
    // This used to fail because no DimName impl exists for this dimension
    use symplex::units::constants;
    let ctx = symplex::prelude::Context::new();
    let kb = constants::boltzmann_constant(&ctx);
    let s = format!("{}", kb);
    assert!(s.contains("k_B"), "should display the constant name: {s}");
}

#[test]
fn qty_debug_exotic_dim() {
    use symplex::units::constants;
    let ctx = symplex::prelude::Context::new();
    let kb = constants::boltzmann_constant(&ctx);
    let s = format!("{:?}", kb);
    assert!(s.contains("k_B"), "debug should contain constant name: {s}");
}

#[test]
fn named_display_keeps_suffix() {
    let ctx = Context::new();
    let f = Force::constant(&ctx, 98);
    let s = format!("{}", f);
    assert!(
        s.contains("[N]"),
        "Force display should keep unit suffix: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// New Mul chain entries
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn momentum_times_velocity_is_energy() {
    let ctx = Context::new();
    let p = Momentum::constant(&ctx, 10);
    let v = Velocity::constant(&ctx, 3);
    let e = symplex::dim!(ctx, Energy: p * v);
    assert_eq!(e.eval_f64().unwrap(), 30.0);
}

#[test]
fn velocity_times_momentum_is_energy() {
    let ctx = Context::new();
    let v = Velocity::constant(&ctx, 3);
    let p = Momentum::constant(&ctx, 10);
    let e = symplex::dim!(ctx, Energy: v * p);
    assert_eq!(e.eval_f64().unwrap(), 30.0);
}

#[test]
fn e_mc_squared_typed_chain() {
    let ctx = symplex::prelude::Context::new();
    let m = Mass::constant(&ctx, 1);
    use symplex::units::constants;
    let c = constants::speed_of_light(&ctx);
    // Mass × Velocity × Velocity = Energy (via dim! macro)
    let e = symplex::dim!(ctx, Energy: m * c * c);
    let val = e.eval_f64().unwrap();
    let expected = 299792458.0_f64 * 299792458.0;
    assert!((val - expected).abs() / expected < 1e-10);
}

#[test]
fn angular_momentum_times_angvel_is_energy() {
    let ctx = Context::new();
    let l = AngularMomentum::constant(&ctx, 5);
    let w = AngularVelocity::constant(&ctx, 4);
    let e = symplex::dim!(ctx, Energy: l * w);
    assert_eq!(e.eval_f64().unwrap(), 20.0);
}

#[test]
fn half_i_omega_squared_chain() {
    let ctx = Context::new();
    // T = ½Iω²: MoI×AngVel×AngVel=Energy (via dim! macro)
    let i_moi = MomentOfInertia::constant(&ctx, 2);
    let w = AngularVelocity::constant(&ctx, 3);
    let ke = symplex::dim!(ctx, Energy: i_moi * w * w);
    // Result should be 2*3*3 = 18 (without the ½)
    assert_eq!(ke.eval_f64().unwrap(), 18.0);
}

#[test]
fn force_times_time_is_momentum() {
    let ctx = Context::new();
    let f = Force::constant(&ctx, 50);
    let t = Time::constant(&ctx, 2);
    let j = symplex::dim!(ctx, Momentum: f * t);
    assert_eq!(j.eval_f64().unwrap(), 100.0);
}

#[test]
fn capacitance_times_voltage_is_charge() {
    let ctx = Context::new();
    let cap = Capacitance::rational(&ctx, 1, 1000); // 1 mF
    let v = Voltage::constant(&ctx, 5);
    let q = symplex::dim!(ctx, Charge: cap * v);
    let val = q.eval_f64().unwrap();
    assert!((val - 0.005).abs() < 1e-10);
}

#[test]
fn pressure_times_area_is_force() {
    let ctx = Context::new();
    let p = Pressure::constant(&ctx, 100000); // 100 kPa
    let a = Area::constant(&ctx, 2);
    let f = symplex::dim!(ctx, Force: p * a);
    assert_eq!(f.eval_f64().unwrap(), 200000.0);
}

#[test]
fn pressure_times_volume_is_energy() {
    let ctx = Context::new();
    let p = Pressure::constant(&ctx, 101325); // 1 atm
    let v = Volume::constant(&ctx, 1);
    let e = symplex::dim!(ctx, Energy: p * v);
    assert_eq!(e.eval_f64().unwrap(), 101325.0);
}
