//! Engineering Units — motor design, imperial conversions, and code generation.
//!
//! Demonstrates a practical engineering workflow with compile-time dimensional
//! analysis, exact imperial unit conversions, and code generation with uom types.
//!
//! Run with: cargo run --example units_engineering

use symplex::prelude::*;
use symplex::units::*;
use symplex::units::constants;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("   Engineering Units — Motor Design & Imperial Conversions");
    println!("═══════════════════════════════════════════════════════════════\n");

    section_1_motor_specs();
    section_2_power_analysis();
    section_3_imperial_conversions();
    section_4_unit_conversion_showcase();
    section_5_codegen_with_uom();

    println!("═══════════════════════════════════════════════════════════════");
    println!("   ✓ All computations dimension-checked. All conversions exact.");
    println!("═══════════════════════════════════════════════════════════════");
}

fn section_1_motor_specs() {
    println!("── 1. DC Motor Specification ──\n");

    // Define motor parameters with units
    let v_rated = Voltage::constant(24);          // 24 V
    let i_rated = Current::constant(10);          // 10 A
    let r_wind = Resistance::rational(12, 10);    // 1.2 Ω
    let l_wind = Inductance::rational(5, 1000);   // 5 mH

    println!("  Rated voltage:  {}", v_rated);
    println!("  Rated current:  {}", i_rated);
    println!("  Winding R:      {}", r_wind);
    println!("  Winding L:      {}", l_wind);

    // Ohm's law: V = IR (compile-time dimension check)
    let v_drop = symplex::dim!(Voltage: i_rated * r_wind);
    println!("\n  Voltage drop across winding: V = IR = {}", v_drop);

    // Back-EMF voltage
    let v_bemf: Voltage = &v_rated - &v_drop;
    println!("  Back-EMF voltage: V_bemf = V_rated - IR = {}", v_bemf);
    println!();
}

fn section_2_power_analysis() {
    println!("── 2. Power Analysis ──\n");

    let v = Voltage::constant(24);
    let i = Current::constant(10);
    let r = Resistance::rational(12, 10);

    // Input power: P_in = I × V  (Current × Voltage → Power, dimension-checked via dim!)
    let p_in = symplex::dim!(Power: i * v);
    println!("  P_in = V × I = {}", p_in);

    // Copper losses: P_loss = I²R = I × (I × R) = I × V_drop
    let v_drop = symplex::dim!(Voltage: i * r);
    let p_loss = symplex::dim!(Power: i * v_drop);
    println!("  P_loss = I²R = {}", p_loss);

    // Mechanical power: P_mech = P_in - P_loss
    let p_mech: Power = &p_in - &p_loss;
    println!("  P_mech = P_in - P_loss = {}", p_mech);

    // Efficiency
    let p_mech_f64 = p_mech.eval_f64().unwrap();
    let p_in_f64 = p_in.eval_f64().unwrap();
    println!("\n  Efficiency = P_mech/P_in = {}/{} = {}%",
             p_mech.inner(), p_in.inner(),
             (p_mech_f64 / p_in_f64 * 100.0) as i32);

    // Convert to horsepower (exact!)
    let one = symplex::default_context().int(1);
    let hp_factor = Power::horsepower(&one).eval_f64().unwrap();
    println!("  P_mech = {:.3} hp", p_mech_f64 / hp_factor);
    println!();
}

fn section_3_imperial_conversions() {
    println!("── 3. Imperial Conversions (all exact!) ──\n");

    let one = symplex::default_context().int(1);

    // Every conversion is an exact rational — no floating point
    println!("  Force:");
    println!("    1 lbf = {} N", Force::pound_force(&one).eval());
    println!("    1 kgf = {} N", Force::kilogram_force(&one).eval());

    println!("  Pressure:");
    println!("    1 psi  = {} Pa", Pressure::psi(&one).eval());
    println!("    1 torr = {} Pa", Pressure::torr(&one).eval());

    println!("  Energy:");
    println!("    1 BTU  = {} J", Energy::btu(&one).eval());
    println!("    1 cal  = {} J", Energy::calories(&one).eval());
    println!("    1 ft·lbf = {} J", Energy::foot_pounds(&one).eval());

    println!("  Volume:");
    println!("    1 US gal = {} m³", Volume::us_gallons(&one).eval());
    println!("    1 US gal = {:.6} L",
             Volume::us_gallons(&one).eval_f64().unwrap() * 1000.0);
    println!("    1 imp gal = {} m³", Volume::imperial_gallons(&one).eval());

    println!("  Speed:");
    println!("    60 mph = {} m/s",
             Velocity::miles_per_hour(&symplex::default_context().int(60)).eval());
    println!("    1 knot = {} m/s", Velocity::knots(&one).eval());

    println!("  Mass:");
    println!("    1 slug = {} kg",
             Mass::slugs(&one).eval());
    println!("    1 oz = {} kg", Mass::ounces(&one).eval());

    println!();
}

fn section_4_unit_conversion_showcase() {
    println!("── 4. Real-World Conversions ──\n");

    // Tire pressure: 32 psi → kPa
    let tire_psi = symplex::default_context().int(32);
    let tire_pa = Pressure::psi(&tire_psi);
    println!("  Tire pressure: 32 psi = {:.1} kPa",
             tire_pa.eval_f64().unwrap() / 1000.0);

    // Speed limit: 65 mph → km/h
    let speed_mph = symplex::default_context().int(65);
    let speed_ms = Velocity::miles_per_hour(&speed_mph);
    println!("  Speed limit: 65 mph = {:.1} km/h",
             speed_ms.eval_f64().unwrap() * 3.6);

    // Engine power: 200 hp → kW
    let engine_hp = symplex::default_context().int(200);
    let engine_w = Power::horsepower(&engine_hp);
    println!("  Engine power: 200 hp = {:.1} kW",
             engine_w.eval_f64().unwrap() / 1000.0);

    // Fuel tank: 15 US gallons → liters
    let tank_gal = symplex::default_context().int(15);
    let tank_m3 = Volume::us_gallons(&tank_gal);
    println!("  Fuel tank: 15 gal = {:.1} L",
             tank_m3.eval_f64().unwrap() * 1000.0);

    // Room temperature: 72°F → K
    let temp_f = symplex::default_context().int(72);
    let temp_k = Temperature::from_fahrenheit(&temp_f);
    println!("  Room temp: 72°F = {:.2} K = {:.2}°C",
             temp_k.eval_f64().unwrap(),
             temp_k.eval_f64().unwrap() - 273.15);

    // Standard gravity from physical constants
    let g = constants::standard_gravity();
    println!("\n  Standard gravity: g₀ = {} (physical constant, exact)", g);
    println!("  g₀ = {:.5} m/s²", g.eval_f64().unwrap());

    println!();
}

#[allow(non_snake_case)]
fn section_5_codegen_with_uom() {
    println!("── 5. Code Generation with uom Types ──\n");

    // Build a simple motor torque equation symbolically
    let __ctx = symplex::default_context();
    symplex::syms!(__ctx; V, R_m, Kt);

    // Stall torque: τ = Kt·V/R
    // At stall (ω=0): I_stall = V/R, τ_stall = Kt × V/R
    let tau_stall = expr!(Kt * V / R_m);

    println!("  Stall torque: τ = Kt·V/R = {}", tau_stall);

    // Generate code with uom annotations
    use symplex::matrix::CodegenOptions;
    let opts = CodegenOptions::default()
        .with_uom()
        .param_unit("V", "ElectricPotential")
        .param_unit("R_m", "ElectricalResistance")
        .param_unit("Kt", "Torque")
        .return_unit_type("Torque");

    let code = tau_stall.to_rust_fn_with_options(
        "stall_torque", &["Kt", "V", "R_m"], &opts
    );

    match code {
        Ok(c) => {
            println!("  Generated function with uom types:");
            for line in c.lines().take(10) {
                println!("    {}", line);
            }
            if c.lines().count() > 10 {
                println!("    ...");
            }
        }
        Err(e) => println!("  Codegen: {:?}", e),
    }
    println!();
}
