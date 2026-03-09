//! Lagrangian mechanics with compile-time dimensional analysis.
//!
//! THE SHOWCASE EXAMPLE: derives equations of motion from energy,
//! with full dimension checking, then generates Rust code.
//!
//! Run with: cargo run --example units_lagrangian

use symplex::prelude::*;
use symplex::units::*;
use symplex::units::constants;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("   Symplex: Lagrangian Mechanics — The Showcase Example");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Section 1: Simple Pendulum
    //   T = ½ml²θ̇²     (kinetic energy)
    //   V = mgl(1−cosθ)  (potential energy)
    //   L = T − V        (Lagrangian)
    //   ∂L/∂θ̇ → angular momentum
    //   ∂L/∂θ → torque (equation of motion)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("── Simple Pendulum ──");

    // Raw Ex variables for use inside expr! — the most ergonomic way
    // to build complex symbolic formulas.
    let ctx = Context::new();
    symplex::syms!(ctx; m, l, g, theta, theta_dot);

    // Typed variables for DiffWrt — the compiler tracks dimensions
    // and verifies that differentiation produces the correct output type.
    let theta_var = Angle::symbol("theta");
    let theta_dot_var = AngularVelocity::symbol("theta_dot");

    // ── Build energies with expr! ──
    // Kinetic energy: T = ½ml²θ̇²
    let ke = Energy::from_ex(expr!(ctx, 1/2 * m * l^2 * theta_dot^2));
    println!("  T = {}", ke);

    // Potential energy: V = mgl(1 − cos θ)
    let pe = Energy::from_ex(expr!(ctx, m * g * l * (1 - cos(theta))));
    println!("  V = {}", pe);

    // ── Lagrangian: Energy − Energy = Energy (dimension checked!) ──
    let lagrangian: Energy = &ke - &pe;
    println!("  L = T − V = {}", lagrangian);

    // ── ∂L/∂θ̇ → AngularMomentum (typed DiffWrt!) ──
    // The compiler verifies: d(Energy)/d(AngularVelocity) = AngularMomentum
    let dl_dthetadot: AngularMomentum = lagrangian.diff_wrt(&theta_dot_var);
    println!("  ∂L/∂θ̇ = {}", dl_dthetadot);

    // ── ∂L/∂θ → Torque (typed DiffWrt!) ──
    // The compiler verifies: d(Energy)/d(Angle) = Torque
    let dl_dtheta: Torque = lagrangian.diff_wrt(&theta_var);
    println!("  ∂L/∂θ  = {}", dl_dtheta);

    // Simplify the torque expression
    let dl_dtheta_simplified = dl_dtheta.clone().simplify();
    println!("  ∂L/∂θ simplified = {}", dl_dtheta_simplified);

    // The Euler–Lagrange equation of motion is:
    //   d/dt(∂L/∂θ̇) − ∂L/∂θ = 0
    // which yields:  ml²θ̈ = −mgl·sin(θ)
    println!("  Euler–Lagrange: d/dt(∂L/∂θ̇) − ∂L/∂θ = 0");
    println!("  → ml²θ̈ = ∂L/∂θ");

    // Using the physical constant for g:
    let g_const = constants::standard_gravity();
    println!("  g₀ = {} (physical constant, exact)", g_const);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Section 2: Spring-Mass-Damper
    //   F = −kx − cv          (force equation)
    //   a = F/m               (Newton's second law)
    //   PE = ½kx²             (elastic potential energy)
    //   F_from_PE = −dPE/dx   (force derived from potential)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Spring-Mass-Damper ──");

    // Named typed variables — compile-time dimension checking for
    // every multiplication, division, and addition.
    let k = Stiffness::symbol("k");
    let x = Length::symbol("x");
    let c = Damping::symbol("c");
    let v = Velocity::symbol("v");
    let mass = Mass::symbol("m");

    // Spring force: Stiffness × Length → Force (compile-time verified)
    let f_spring = symplex::dim!(ctx, Force: -(k * x));
    println!("  F_spring = −kx = {}", f_spring);

    // Damping force: Damping × Velocity → Force (compile-time verified)
    let f_damper = symplex::dim!(ctx, Force: -(c * v));
    println!("  F_damper = −cv = {}", f_damper);

    // Total force: Force + Force → Force (same-type addition)
    let f_total: Force = &f_spring + &f_damper;
    println!("  F_total = {}", f_total);

    // Newton's second law: Force / Mass → Acceleration
    let accel = symplex::dim!(ctx, Acceleration: f_total / mass);
    println!("  a = F/m = {}", accel);

    // ── Potential energy approach: PE = ½kx² ──
    // Use expr! for the formula, then derive force via differentiation
    let ctx = Context::new();
    symplex::syms!(ctx; k_var, x_var);
    let spring_pe = Energy::from_ex(expr!(ctx, 1/2 * k_var * x_var^2));
    println!("\n  PE = ½kx² = {}", spring_pe);

    // Force from potential: F = −dPE/dx
    // Energy.diff_wrt(Length) → Force, then negate
    let x_typed = Length::symbol("x_var");
    let f_from_pe: Force = spring_pe.diff_wrt(&x_typed);
    let f_from_pe_neg: Force = -f_from_pe;
    println!("  F = −dPE/dx = {}", f_from_pe_neg);
    println!("  ✓ This equals −kx (force from Hooke's law)");

    // ── Energy conservation check ──
    // KE = ½mv² using expr!
    let ctx = Context::new();
    symplex::syms!(ctx; m_raw, v_raw);
    let spring_ke = Energy::from_ex(expr!(ctx, 1/2 * m_raw * v_raw^2));
    println!("\n  KE = ½mv² = {}", spring_ke);

    // Total energy: Energy + Energy = Energy
    let total_energy: Energy = &spring_ke + &spring_pe;
    println!("  E_total = KE + PE = {}", total_energy);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Section 3: Numerical Evaluation
    //   Substitute concrete values into the pendulum expressions and
    //   compute numerical answers.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Numerical Evaluation ──");

    // Pendulum parameters: m = 1 kg, l = 0.5 m, g = 9.81 m/s², θ = 0.1 rad
    println!("  Parameters: m=1 kg, l=0.5 m, g=9.81 m/s², θ=0.1 rad");

    // Evaluate the angular momentum ∂L/∂θ̇ at θ̇ = 2 rad/s
    let ang_mom_num = dl_dthetadot.clone()
        .subs(&m, &ctx.int(1))
        .subs(&l, &ctx.rational(1, 2))
        .subs(&theta_dot, &ctx.int(2))
        .eval();
    println!("  ∂L/∂θ̇(m=1, l=0.5, θ̇=2) = {}", ang_mom_num);

    // Evaluate the torque ∂L/∂θ at θ = 0.1 rad
    let torque_num = dl_dtheta.clone()
        .subs(&m, &ctx.int(1))
        .subs(&g, &ctx.rational(981, 100))
        .subs(&l, &ctx.rational(1, 2))
        .subs(&theta, &ctx.rational(1, 10))
        .eval();
    println!("  ∂L/∂θ(m=1, g=9.81, l=0.5, θ=0.1) = {}", torque_num);

    // Use eval_f64_with for a quick numeric answer on KE
    let ke_f64 = ke
        .eval_f64_with(&[(&m, 1), (&l, 1), (&theta_dot, 3)])
        .unwrap();
    println!("  T(m=1, l=1, θ̇=3) = {:.4} J (f64)", ke_f64);

    // Evaluate PE at various angles
    println!("\n  PE at various angles (m=1, g=10, l=1):");
    for angle_deg in [0, 15, 30, 45, 60, 90] {
        let pe_val = pe.clone()
            .subs(&m, &ctx.int(1))
            .subs(&g, &ctx.int(10))
            .subs(&l, &ctx.int(1))
            .subs(&theta, &Angle::degrees(&ctx.int(angle_deg)).into_inner())
            .eval();
        // Use eval_f64 for a readable number
        if let Ok(f) = pe_val.eval_f64() {
            println!("    θ = {:>3}° → V ≈ {:.4} J", angle_deg, f);
        } else {
            println!("    θ = {:>3}° → V = {}", angle_deg, pe_val);
        }
    }

    // Spring-mass numerical check: PE = ½kx²
    // k = 100 N/m, x = 0.2 m → PE = ½·100·0.04 = 2 J
    let spring_pe_num = spring_pe.clone()
        .subs(&k_var, &ctx.int(100))
        .subs(&x_var, &ctx.rational(1, 5))
        .eval();
    println!("\n  Spring PE(k=100, x=0.2) = {} (expect 2 J)", spring_pe_num);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Section 4: Code Generation
    //   Generate optimized Rust code from the symbolic torque expression.
    //   THIS IS THE DEMO: physics → symbolic math → units → code.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Code Generation ──");
    println!("  Generating Rust code from the pendulum torque expression...\n");

    // The torque expression ∂L/∂θ is the equation of motion for the
    // pendulum.  We can turn it directly into optimized Rust code!
    //
    // .inner() drops the dimension wrapper, giving us the raw Ex.
    // .to_rust_fn() performs Common Subexpression Elimination (CSE)
    // and emits a clean Rust function.
    let torque_code = dl_dtheta
        .inner()
        .to_rust_fn("pendulum_torque", &["m", "l", "g", "theta"])
        .unwrap();
    println!("{torque_code}");

    // Also generate code for the kinetic energy
    println!("  --- Kinetic energy function ---\n");
    let ke_code = ke
        .inner()
        .to_rust_fn("pendulum_kinetic_energy", &["m", "l", "theta_dot"])
        .unwrap();
    println!("{ke_code}");

    // And the spring potential energy
    println!("  --- Spring PE function ---\n");
    let spring_code = spring_pe
        .inner()
        .to_rust_fn("spring_potential_energy", &["k_var", "x_var"])
        .unwrap();
    println!("{spring_code}");

    // ── Summary ────────────────────────────────────────────────────────
    println!("  The workflow:");
    println!("    1. Write physics formulas with expr! (symbolic)");
    println!("    2. Wrap in dimension types (compile-time safety)");
    println!("    3. Differentiate with DiffWrt (typed calculus)");
    println!("    4. Substitute & evaluate (numerical answers)");
    println!("    5. Generate Rust code (production deployment)");
    println!("  All with the compiler verifying dimensions at every step!");

    println!("\n✓ All done!");
}
