//! Physical constants — symbolic display with exact evaluation.
//!
//! symplex physical constants display as their standard symbols (c, h, k_B)
//! but evaluate to their exact SI values. No floating-point approximation.
//!
//! Run with: cargo run --example physical_constants

use symplex::prelude::*;
use symplex::units::*;
use symplex::units::constants;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("   Physical Constants — Symbolic Display, Exact Evaluation");
    println!("═══════════════════════════════════════════════════════════════\n");

    section_1_symbolic_display();
    section_2_e_mc_squared();
    section_3_photon_energy();
    section_4_thermal_energy();
    section_5_gravity();
    section_6_constants_with_calculus();
    section_7_dimensional_checking();

    println!("═══════════════════════════════════════════════════════════════");
    println!("   ✓ All physical constants are exact — no approximation!");
    println!("═══════════════════════════════════════════════════════════════");
}

fn section_1_symbolic_display() {
    println!("── 1. Constants Display as Symbols ──\n");

    let c = constants::speed_of_light();
    let h = constants::planck_constant();
    let kb = constants::boltzmann_constant();
    let g = constants::standard_gravity();

    println!("  Speed of light:    {} (displays as symbol)", c);
    println!("  Planck constant:   {} (displays as symbol)", h);
    println!("  Boltzmann:         {} (displays as symbol)", kb.inner());
    println!("  Standard gravity:  {} (displays as symbol)", g);

    println!("\n  Each evaluates to its exact SI value:");
    println!("  c     = {:.10} m/s", c.eval_f64().unwrap());
    println!("  h     = {:.6e} J·s", h.eval_f64().unwrap());
    println!("  k_B   = {:.6e} J/K", kb.eval_f64().unwrap());
    println!("  g₀    = {:.5} m/s²", g.eval_f64().unwrap());
    println!();
}

fn section_2_e_mc_squared() {
    let ctx = Context::new();
    println!("── 2. E = mc² — Rest Energy ──\n");

    let c = constants::speed_of_light();
    let m = Mass::symbol("m");

    // E = mc² — the expression stays symbolic
    let energy = symplex::dim!(Energy: m * c * c);
    println!("  E = mc² = {}", energy);
    println!("  (Notice: 'c' not '299792458')\n");

    // Evaluate for 1 kg
    let e_1kg = energy.subs(&m, &ctx.int(1)).eval_f64().unwrap();
    println!("  E(m = 1 kg) = {:.6e} J", e_1kg);
    println!("              = {:.6e} GJ", e_1kg / 1e9);
    println!("  That's ~25 million kilowatt-hours from 1 kg of matter!");
    println!();
}

fn section_3_photon_energy() {
    println!("── 3. E = hf — Photon Energy ──\n");

    let h = constants::planck_constant();
    let c = constants::speed_of_light();

    // Energy of a photon: E = hf
    println!("  E = h·f (symbolic): h*f");

    // For green light: λ = 500 nm, f = c/λ
    // E = hc/λ
    let hc = h.inner() * c.inner();
    println!("  h·c = {} (product of two constants)", hc);

    // Evaluate for λ = 500 nm = 5e-7 m
    let lambda_val = 500e-9_f64;
    let e_photon = h.eval_f64().unwrap() * c.eval_f64().unwrap() / lambda_val;
    println!("  E(λ=500nm) = {:.4e} J", e_photon);
    println!("             = {:.4} eV", e_photon / 1.602176634e-19);
    println!("  (Green light photons carry about 2.48 eV)");
    println!();
}

fn section_4_thermal_energy() {
    println!("── 4. E = k_B·T — Thermal Energy ──\n");

    let kb = constants::boltzmann_constant();

    // At room temperature: T = 300 K
    let t_room = 300.0_f64;
    let e_thermal = kb.eval_f64().unwrap() * t_room;
    println!("  k_B·T at 300 K = {:.4e} J", e_thermal);
    println!("                 = {:.4} meV", e_thermal / 1.602176634e-19 * 1000.0);
    println!("  (Thermal energy at room temperature ≈ 25.9 meV)");
    println!();
}

fn section_5_gravity() {
    println!("── 5. F = Gm₁m₂/r² — Gravitational Force ──\n");

    let ctx = Context::new();
    symplex::syms!(ctx; m1, m2, r);
    let g_const = constants::gravitational_constant();

    // Newton's law of gravitation
    let force_expr = g_const.inner() * &m1 * &m2 / &r.powi(2);
    println!("  F = G·m₁·m₂/r² = {}", force_expr);

    // Earth-Moon system
    let m_earth = 5.972e24_f64;
    let m_moon = 7.342e22_f64;
    let r_em = 384_400_000.0_f64; // meters
    let g_val = g_const.eval_f64().unwrap();
    let f_em = g_val * m_earth * m_moon / (r_em * r_em);
    println!("  Earth-Moon: F = {:.3e} N", f_em);
    println!("  (About 1.98 × 10²⁰ newtons)");
    println!();
}

fn section_6_constants_with_calculus() {
    println!("── 6. Constants and Calculus ──\n");

    let c = constants::speed_of_light();
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    // d/dx(c·x) = c (constant preserved through differentiation)
    let cx = c.inner() * &x;
    let dcx_dx = cx.diff(&x);
    println!("  d/dx(c·x) = {}", dcx_dx);
    println!("  (The constant c is preserved, not expanded to 299792458)\n");

    // d/dx(c) = 0
    let dc_dx = c.diff(&x);
    println!("  d/dx(c) = {}", dc_dx);
    println!("  (Derivative of a constant is zero)");
    println!();
}

fn section_7_dimensional_checking() {
    let ctx = Context::new();
    println!("── 7. Constants Carry Dimensions ──\n");

    let c = constants::speed_of_light();
    let g = constants::standard_gravity();

    println!("  speed_of_light()   → {} (Velocity)", Velocity::dim_name_str());
    println!("  planck_constant()  → {} (AngularMomentum)", AngularMomentum::dim_name_str());
    println!("  standard_gravity() → {} (Acceleration)", Acceleration::dim_name_str());

    // mc² type-checks as Energy
    let m = Mass::symbol("m");
    let _energy = symplex::dim!(Energy: m * c * c);
    println!("\n  m·c² type-checks as Energy ✓");

    // mg type-checks as Force
    let _weight = symplex::dim!(Force: m * g);
    println!("  m·g₀ type-checks as Force ✓");

    println!();
}
