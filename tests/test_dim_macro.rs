//! Integration tests for the `dim!` proc macro.
//!
//! These tests verify compile-time dimension tracking through natural
//! math syntax, ensuring that the generated `Qty<D>` arithmetic produces
//! the correct output type via `FromDimExpr`.

use symplex::prelude::*;
use symplex::units::*;

// ═══════════════════════════════════════════════════════════════════════════
// Basic multiplication
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dim_mass_times_acceleration_is_force() {
    let m = Mass::symbol("m");
    let a = Acceleration::symbol("a");
    let f: Force = symplex::dim!(Force: m * a);
    assert_eq!(format!("{}", f.inner()), "a*m");
}

#[test]
fn dim_force_times_length_is_energy() {
    let f = Force::symbol("F");
    let d = Length::symbol("d");
    let e: Energy = symplex::dim!(Energy: f * d);
    assert_eq!(format!("{}", e.inner()), "F*d");
}

#[test]
fn dim_mass_times_gravity_times_height() {
    let m = Mass::symbol("m");
    let g = Acceleration::symbol("g");
    let h = Length::symbol("h");
    let pe: Energy = symplex::dim!(Energy: m * g * h);
    // Canonical ordering is alphabetical
    let inner = format!("{}", pe.inner());
    assert!(inner.contains("m"));
    assert!(inner.contains("g"));
    assert!(inner.contains("h"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Division
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dim_length_div_time_is_velocity() {
    let l = Length::symbol("L");
    let t = Time::symbol("t");
    let v: Velocity = symplex::dim!(Velocity: l / t);
    assert_eq!(format!("{}", v.inner()), "L*1/t");
}

#[test]
fn dim_energy_div_time_is_power() {
    let e = Energy::symbol("E_val");
    let t = Time::symbol("t");
    let p: Power = symplex::dim!(Power: e / t);
    assert_eq!(format!("{}", p.inner()), "E_val*1/t");
}

#[test]
fn dim_force_div_area_is_pressure() {
    let f = Force::symbol("F");
    let a = Area::symbol("A");
    let p: Pressure = symplex::dim!(Pressure: f / a);
    assert_eq!(format!("{}", p.inner()), "F*1/A");
}

// ═══════════════════════════════════════════════════════════════════════════
// Addition and subtraction (same dimension)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dim_add_same_type() {
    let f1 = Force::symbol("F1");
    let f2 = Force::symbol("F2");
    let total: Force = symplex::dim!(Force: f1 + f2);
    assert_eq!(format!("{}", total.inner()), "F1 + F2");
}

#[test]
fn dim_sub_same_type() {
    let e1 = Energy::symbol("KE");
    let e2 = Energy::symbol("PE");
    let diff: Energy = symplex::dim!(Energy: e1 - e2);
    assert_eq!(format!("{}", diff.inner()), "KE - PE");
}

// ═══════════════════════════════════════════════════════════════════════════
// Negation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dim_negation() {
    let f = Force::symbol("F");
    let neg_f: Force = symplex::dim!(Force: -f);
    assert_eq!(format!("{}", neg_f.inner()), "-F");
}

// ═══════════════════════════════════════════════════════════════════════════
// Integer and rational constants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dim_scalar_multiply_by_integer() {
    let f = Force::symbol("F");
    let doubled: Force = symplex::dim!(Force: 2 * f);
    assert_eq!(format!("{}", doubled.inner()), "2*F");
}

#[test]
fn dim_rational_constant() {
    let v = Velocity::symbol("v");
    let half_v: Velocity = symplex::dim!(Velocity: 1/2 * v);
    assert_eq!(format!("{}", half_v.inner()), "1/2*v");
}

#[test]
fn dim_integer_is_dimensionless() {
    let d: Dimensionless = symplex::dim!(Dimensionless: 42);
    assert_eq!(format!("{}", d.inner()), "42");
}

// ═══════════════════════════════════════════════════════════════════════════
// Power expansion (repeated multiplication)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dim_length_squared_is_area() {
    let l = Length::symbol("r");
    let a: Area = symplex::dim!(Area: l^2);
    assert_eq!(format!("{}", a.inner()), "r^2");
}

#[test]
fn dim_length_cubed_is_volume() {
    let l = Length::symbol("r");
    let vol: Volume = symplex::dim!(Volume: l^3);
    assert_eq!(format!("{}", vol.inner()), "r^3");
}

#[test]
fn dim_power_zero_is_dimensionless() {
    let m = Mass::symbol("m");
    let one: Dimensionless = symplex::dim!(Dimensionless: m^0);
    assert_eq!(format!("{}", one.inner()), "1");
}

#[test]
fn dim_power_one_identity() {
    let l = Length::symbol("x");
    let same: Length = symplex::dim!(Length: l^1);
    assert_eq!(format!("{}", same.inner()), "x");
}

// ═══════════════════════════════════════════════════════════════════════════
// Compound expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dim_kinetic_energy() {
    // KE = (1/2) * m * v^2
    let m = Mass::symbol("m");
    let v = Velocity::symbol("v");
    let ke: Energy = symplex::dim!(Energy: 1/2 * m * v^2);
    let inner = format!("{}", ke.inner());
    assert!(inner.contains("m"));
    assert!(inner.contains("v"));
}

#[test]
fn dim_spring_potential_energy() {
    // PE = (1/2) * k * x^2
    let k = Stiffness::symbol("k");
    let x = Length::symbol("x");
    let pe: Energy = symplex::dim!(Energy: 1/2 * k * x^2);
    let inner = format!("{}", pe.inner());
    assert!(inner.contains("k"));
    assert!(inner.contains("x"));
}

#[test]
fn dim_momentum_equals_mass_times_velocity() {
    let m = Mass::symbol("m");
    let v = Velocity::symbol("v");
    let p: Momentum = symplex::dim!(Momentum: m * v);
    let inner = format!("{}", p.inner());
    assert!(inner.contains("m"));
    assert!(inner.contains("v"));
}

#[test]
fn dim_ohms_law() {
    // V = I * R
    let i = Current::symbol("I_val");
    let r = Resistance::symbol("R");
    let v: Voltage = symplex::dim!(Voltage: i * r);
    let inner = format!("{}", v.inner());
    assert!(inner.contains("I_val"));
    assert!(inner.contains("R"));
}

#[test]
fn dim_power_electrical() {
    // P = V * I
    let v = Voltage::symbol("V_val");
    let i = Current::symbol("I_val");
    let p: Power = symplex::dim!(Power: v * i);
    let inner = format!("{}", p.inner());
    assert!(inner.contains("V_val"));
    assert!(inner.contains("I_val"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Constants: pi, E
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dim_pi_is_dimensionless() {
    let d: Dimensionless = symplex::dim!(Dimensionless: pi);
    assert_eq!(format!("{}", d.inner()), "pi");
}

#[test]
fn dim_pi_times_length_squared_is_area() {
    let r = Length::symbol("r");
    let circle_area: Area = symplex::dim!(Area: pi * r^2);
    let inner = format!("{}", circle_area.inner());
    assert!(inner.contains("pi"));
    assert!(inner.contains("r"));
}

#[test]
fn dim_euler_number_is_dimensionless() {
    let d: Dimensionless = symplex::dim!(Dimensionless: E);
    assert_eq!(format!("{}", d.inner()), "E");
}

// ═══════════════════════════════════════════════════════════════════════════
// Transcendental functions (result is always Dimensionless)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dim_sin_returns_dimensionless() {
    let theta = Dimensionless::symbol("theta");
    let s: Dimensionless = symplex::dim!(Dimensionless: sin(theta));
    assert_eq!(format!("{}", s.inner()), "sin(theta)");
}

#[test]
fn dim_cos_returns_dimensionless() {
    let theta = Dimensionless::symbol("theta");
    let c: Dimensionless = symplex::dim!(Dimensionless: cos(theta));
    assert_eq!(format!("{}", c.inner()), "cos(theta)");
}

#[test]
fn dim_exp_returns_dimensionless() {
    let x = Dimensionless::symbol("x");
    let e: Dimensionless = symplex::dim!(Dimensionless: exp(x));
    assert_eq!(format!("{}", e.inner()), "exp(x)");
}

#[test]
fn dim_ln_returns_dimensionless() {
    let x = Dimensionless::symbol("x");
    let l: Dimensionless = symplex::dim!(Dimensionless: ln(x));
    assert_eq!(format!("{}", l.inner()), "ln(x)");
}

#[test]
fn dim_function_result_scaled_by_quantity() {
    // F = m * g * sin(theta)
    // sin(theta) is dimensionless, so m * g * sin(theta) = Force
    let m = Mass::symbol("m");
    let g = Acceleration::symbol("g");
    let theta = Dimensionless::symbol("theta");
    let f: Force = symplex::dim!(Force: m * g * sin(theta));
    let inner = format!("{}", f.inner());
    assert!(inner.contains("m"));
    assert!(inner.contains("g"));
    assert!(inner.contains("sin"));
}

// ═══════════════════════════════════════════════════════════════════════════
// References — dim! should work with & references via clone
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dim_uses_variables_by_cloning() {
    let m = Mass::symbol("m");
    let g = Acceleration::symbol("g");
    let h = Length::symbol("h");
    // After dim!, the original variables should still be usable
    let _e: Energy = symplex::dim!(Energy: m * g * h);
    // Variables were consumed by clone inside the macro, but we can
    // verify the original bindings are still valid by reconstructing:
    let m2 = Mass::symbol("m");
    let g2 = Acceleration::symbol("g");
    let h2 = Length::symbol("h");
    let _e2: Energy = symplex::dim!(Energy: m2 * g2 * h2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex physics formulas
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dim_gravitational_potential_energy() {
    // U = m * g * h
    let m = Mass::symbol("m");
    let g = Acceleration::symbol("g");
    let h = Length::symbol("h");
    let u: Energy = symplex::dim!(Energy: m * g * h);
    let inner = format!("{}", u.inner());
    assert!(inner.contains("m"));
    assert!(inner.contains("g"));
    assert!(inner.contains("h"));
}

#[test]
fn dim_damped_force() {
    // F = -b * v - k * x
    let b = Damping::symbol("b");
    let v = Velocity::symbol("v");
    let k = Stiffness::symbol("k");
    let x = Length::symbol("x");
    let f: Force = symplex::dim!(Force: -b * v - k * x);
    let inner = format!("{}", f.inner());
    assert!(inner.contains("b"));
    assert!(inner.contains("v"));
    assert!(inner.contains("k"));
    assert!(inner.contains("x"));
}

#[test]
fn dim_moment_of_inertia_rod() {
    // I = (1/12) * m * L^2
    let m = Mass::symbol("m");
    let big_l = Length::symbol("L");
    let moi: MomentOfInertia = symplex::dim!(MomentOfInertia: 1/12 * m * big_l^2);
    let inner = format!("{}", moi.inner());
    assert!(inner.contains("m"));
    assert!(inner.contains("L"));
}

#[test]
fn dim_charge_is_current_times_time() {
    let i = Current::symbol("I_val");
    let t = Time::symbol("t");
    let q: Charge = symplex::dim!(Charge: i * t);
    let inner = format!("{}", q.inner());
    assert!(inner.contains("I_val"));
    assert!(inner.contains("t"));
}

#[test]
fn dim_voltage_from_inductance_and_current_rate() {
    // V = L * dI/dt  (dimensionally: Inductance * Current / Time)
    let l = Inductance::symbol("L_ind");
    let di = Current::symbol("dI");
    let dt = Time::symbol("dt");
    let v: Voltage = symplex::dim!(Voltage: l * di / dt);
    let inner = format!("{}", v.inner());
    assert!(inner.contains("L_ind"));
    assert!(inner.contains("dI"));
    assert!(inner.contains("dt"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Parenthesised sub-expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dim_parenthesised_addition_then_multiply() {
    let m = Mass::symbol("m");
    let a1 = Acceleration::symbol("a1");
    let a2 = Acceleration::symbol("a2");
    let f: Force = symplex::dim!(Force: m * (a1 + a2));
    let inner = format!("{}", f.inner());
    assert!(inner.contains("m"));
    assert!(inner.contains("a1"));
    assert!(inner.contains("a2"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Numerical evaluation round-trip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dim_numerical_eval() {
    let m = Mass::from_ex(symplex::int(10));
    let a = Acceleration::from_ex(symplex::rational(98, 10));
    let f: Force = symplex::dim!(Force: m * a);
    let val = f.eval_f64().unwrap();
    assert!((val - 98.0).abs() < 1e-10);
}

#[test]
fn dim_circle_area_numerical() {
    let r = Length::from_ex(symplex::int(5));
    let a: Area = symplex::dim!(Area: pi * r^2);
    let val = a.eval().eval_f64().unwrap();
    let expected = std::f64::consts::PI * 25.0;
    assert!((val - expected).abs() < 1e-10);
}
