//! Integration tests for physical constants with units.

use symplex::prelude::*;
use symplex::units::*;
use symplex::units::constants;

#[test]
fn e_mc_squared_symbolic_display() {
    let m = Mass::symbol("m");
    let c = constants::speed_of_light();
    // Build E = m·c² using raw expressions, then wrap as Energy
    let e = Energy::from_ex(m.inner() * c.inner() * c.inner());
    let display = format!("{}", e.inner());
    assert!(
        display.contains("c"),
        "Should display symbolically with c: {display}"
    );
}

#[test]
fn e_mc_squared_numerical() {
    let c = constants::speed_of_light();
    let m = Mass::constant(1);
    let e = Energy::from_ex(m.inner() * c.inner() * c.inner());
    let val = e.eval_f64().unwrap();
    let expected = 299792458.0_f64.powi(2);
    assert!(
        (val - expected).abs() / expected < 1e-10,
        "E = {val}, expected {expected}"
    );
}

#[test]
fn photon_energy_e_equals_hf() {
    let ctx = symplex::units::si::units_ctx().clone();
    let h = constants::planck_constant();
    // E = hf — h is AngularMomentum (M·L²·T⁻¹), f is Frequency (T⁻¹)
    // AngularMomentum × Frequency → Energy (M·L²·T⁻²)
    let h_qty: Qty<AngularMomentumDim> = h.into();
    let f_qty: Qty<FrequencyDim> = Qty::from_ex(ctx.symbol("f"));
    let e_qty = h_qty * f_qty;
    let e: Energy = e_qty.into();
    let display = format!("{}", e.inner());
    assert!(
        display.contains("h"),
        "E=hf should contain h: {display}"
    );
}

#[test]
fn thermal_energy_kb_t() {
    let ctx = symplex::units::si::units_ctx().clone();
    let kb = constants::boltzmann_constant();
    let t_qty = Qty::<TemperatureDim>::from_ex(ctx.symbol("T"));
    let e_thermal = kb * t_qty;
    // kb × T should have dimension Energy (M·L²·T⁻²·Θ⁻¹ × Θ = M·L²·T⁻²)
    let e: Energy = e_thermal.into();
    let display = format!("{}", e.inner());
    assert!(
        display.contains("k_B"),
        "should contain k_B: {display}"
    );
}

#[test]
fn constant_derivative_is_zero() {
    let ctx = symplex::units::si::units_ctx().clone();
    let c = constants::speed_of_light();
    let h = constants::planck_constant();
    symplex::syms!(ctx; x);
    let dc = c.inner().diff(&x);
    let dh = h.inner().diff(&x);
    assert!(
        dc.is_zero().unwrap_or(false) || format!("{}", dc) == "0",
        "d/dx(c) should be 0, got {dc}"
    );
    assert!(
        dh.is_zero().unwrap_or(false) || format!("{}", dh) == "0",
        "d/dx(h) should be 0, got {dh}"
    );
}

#[test]
fn gravitational_force() {
    let ctx = symplex::units::si::units_ctx().clone();
    let g_const = constants::gravitational_constant();
    symplex::syms!(ctx; m1, m2, r);
    // F = G·m1·m2/r² — raw expression arithmetic (no type-level dimension check)
    let f_expr = g_const.inner() * &m1 * &m2 / &r.powi(2);
    let display = format!("{}", f_expr);
    assert!(
        display.contains("G"),
        "F=Gm1m2/r² should contain G: {display}"
    );
}

#[test]
fn speed_of_light_value() {
    let c = constants::speed_of_light();
    let val = c.eval_f64().unwrap();
    assert!(
        (val - 299_792_458.0).abs() < 1.0,
        "c should be 299792458 m/s, got {val}"
    );
}

#[test]
fn planck_constant_value() {
    let h = constants::planck_constant();
    let val = h.eval_f64().unwrap();
    let expected = 6.62607015e-34;
    assert!(
        (val - expected).abs() / expected < 1e-10,
        "h should be ~6.626e-34, got {val}"
    );
}

#[test]
fn boltzmann_constant_value() {
    let kb = constants::boltzmann_constant();
    let val = kb.eval_f64().unwrap();
    let expected = 1.380649e-23;
    assert!(
        (val - expected).abs() / expected < 1e-10,
        "k_B should be ~1.381e-23, got {val}"
    );
}

#[test]
fn elementary_charge_value() {
    let e = constants::elementary_charge();
    let val = e.eval_f64().unwrap();
    let expected = 1.602176634e-19;
    assert!(
        (val - expected).abs() / expected < 1e-10,
        "e should be ~1.602e-19, got {val}"
    );
}

#[test]
fn standard_gravity_value() {
    let g = constants::standard_gravity();
    let val = g.eval_f64().unwrap();
    assert!(
        (val - 9.80665).abs() < 1e-10,
        "g₀ should be 9.80665, got {val}"
    );
}

#[test]
fn avogadro_constant_value() {
    let na = constants::avogadro_constant();
    let val = na.eval_f64().unwrap();
    let expected = 6.02214076e23;
    assert!(
        (val - expected).abs() / expected < 1e-10,
        "N_A should be ~6.022e23, got {val}"
    );
}

#[test]
fn constant_in_product_preserves_symbol() {
    let ctx = symplex::units::si::units_ctx().clone();
    let c = constants::speed_of_light();
    symplex::syms!(ctx; x);
    let cx = c.inner() * &x;
    let display = format!("{}", cx);
    assert!(
        display.contains("c"),
        "c*x should display with c symbol: {display}"
    );
}

#[test]
fn constant_survives_simplify() {
    let ctx = symplex::units::si::units_ctx().clone();
    let c = constants::speed_of_light();
    symplex::syms!(ctx; x, y);
    let expr = c.inner() * &x + c.inner() * &y;
    let simplified = expr.simplify();
    let display = format!("{}", simplified);
    assert!(
        display.contains("c"),
        "simplify should preserve c: {display}"
    );
}

#[test]
fn constant_diff_in_product() {
    let ctx = symplex::units::si::units_ctx().clone();
    let c = constants::speed_of_light();
    symplex::syms!(ctx; x);
    let cx = c.inner() * &x;
    let d = cx.diff(&x);
    // d/dx(c*x) = c
    let display = format!("{}", d);
    assert!(
        display.contains("c"),
        "d/dx(c*x) should contain c: {display}"
    );
}

#[test]
fn multiple_constants_in_expression() {
    // Build h * c — both should display as their symbols
    let h = constants::planck_constant();
    let c = constants::speed_of_light();
    let product = h.inner() * c.inner();
    let display = format!("{}", product);
    assert!(
        display.contains("h") && display.contains("c"),
        "h*c should contain both symbols: {display}"
    );
}

#[test]
fn gravitational_constant_value() {
    let g = constants::gravitational_constant();
    let val = g.eval_f64().unwrap();
    let expected = 6.67430e-11;
    assert!(
        (val - expected).abs() / expected < 1e-4,
        "G should be ~6.674e-11, got {val}"
    );
}
