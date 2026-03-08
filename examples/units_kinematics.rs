//! Kinematics with compile-time dimensional analysis.
//!
//! Demonstrates: free fall, projectile motion, work-energy theorem.
//! Uses expr! for formulas, DiffWrt for typed calculus.
//!
//! Run with: cargo run --example units_kinematics

use symplex::prelude::*;
use symplex::units::*;
use symplex::units::constants;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("   Symplex: Kinematics with Compile-Time Dimensional Analysis");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Section 1: Free Fall
    //   x(t) = ½gt², v(t) = dx/dt, a(t) = dv/dt
    //   Uses expr! for the formula, DiffWrt for typed derivatives.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("── Free Fall ──");

    // Declare raw Ex variables for use inside expr!
    let __ctx = symplex::default_context();
    symplex::syms!(__ctx; g, t);

    // Typed variables for DiffWrt — the compiler tracks dimensions
    let t_var = Time::symbol("t");

    // Position: x(t) = ½gt² — built ergonomically with expr!
    // from_ex wraps the raw expression in the Length type
    let x_t = Length::from_ex(expr!(1/2 * g * t^2));
    println!("  x(t) = {}", x_t);

    // Velocity: v(t) = dx/dt — typed differentiation!
    // d(Length)/d(Time) → Velocity, verified at compile time
    let v_t: Velocity = x_t.diff_wrt(&t_var);
    println!("  v(t) = dx/dt = {}", v_t);

    // Acceleration: a(t) = dv/dt
    // d(Velocity)/d(Time) → Acceleration, verified at compile time
    let a_t: Acceleration = v_t.diff_wrt(&t_var);
    println!("  a(t) = dv/dt = {}", a_t);

    // Verify: acceleration should equal g (constant gravitational field)
    println!("  ✓ a(t) = {} (should be g)", a_t.inner());

    // Numerical evaluation: x at t=3s with g=9.81 m/s²
    let x_num = x_t.clone()
        .subs(&g, &symplex::default_context().rational(981, 100))
        .subs(&t, &symplex::default_context().int(3))
        .eval();
    println!("  x(t=3, g=9.81) = {} (≈44.145 m)", x_num);

    // Also demonstrate eval_f64_with for quick numeric answers
    let x_f64 = x_t
        .eval_f64_with(&[(&g, 10), (&t, 3)])
        .unwrap();
    println!("  x(t=3, g=10)   = {:.2} m (f64)", x_f64);

    // Using the physical constant for g:
    let g_const = constants::standard_gravity();
    println!("  g₀ = {} (physical constant, exact)", g_const);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Section 2: Projectile Motion
    //   x(t) = v₀·cos(θ)·t,  y(t) = v₀·sin(θ)·t - ½gt²
    //   vx = dx/dt,  vy = dy/dt
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Projectile Motion ──");

    // Raw variables for expr!
    let __ctx = symplex::default_context();
    symplex::syms!(__ctx; v0, theta);
    // (g and t already declared above)

    // Horizontal position: x(t) = v₀·cos(θ)·t
    let x_proj = Length::from_ex(expr!(v0 * cos(theta) * t));
    println!("  x(t) = {}", x_proj);

    // Vertical position: y(t) = v₀·sin(θ)·t - ½gt²
    let y_proj = Length::from_ex(expr!(v0 * sin(theta) * t - 1/2 * g * t^2));
    println!("  y(t) = {}", y_proj);

    // Horizontal velocity: vx = dx/dt (use raw diff + from_ex)
    let vx = Velocity::from_ex(x_proj.diff(&t));
    println!("  vx(t) = dx/dt = {}", vx);

    // Vertical velocity: vy = dy/dt
    let vy = Velocity::from_ex(y_proj.diff(&t));
    println!("  vy(t) = dy/dt = {}", vy);

    // Vertical acceleration: ay = dvy/dt — should be -g
    let ay = Acceleration::from_ex(vy.diff(&t));
    println!("  ay(t) = dvy/dt = {}", ay);
    println!("  ✓ vertical acceleration = {} (should be -g)", ay.inner());

    // Substitute v0=20 m/s, θ=π/4, g=9.81 m/s² and evaluate at several times
    println!("\n  Trajectory (v₀=20 m/s, θ=π/4, g=9.81 m/s²):");
    let pi_over_4 = &symplex::default_context().pi() / 4;
    let g_val = symplex::default_context().rational(981, 100);

    for t_val in [0, 1, 2, 3] {
        let x_val = x_proj.clone()
            .subs(&v0, &symplex::default_context().int(20))
            .subs(&theta, &pi_over_4)
            .subs(&g, &g_val)
            .subs(&t, &symplex::default_context().int(t_val))
            .eval();

        let y_val = y_proj.clone()
            .subs(&v0, &symplex::default_context().int(20))
            .subs(&theta, &pi_over_4)
            .subs(&g, &g_val)
            .subs(&t, &symplex::default_context().int(t_val))
            .eval();

        println!("    t={t_val}s: x = {}, y = {}", x_val, y_val);
    }

    // Quick f64 check at t=1
    let y_f64 = y_proj
        .subs(&v0, &symplex::default_context().int(20))
        .subs(&theta, &pi_over_4)
        .subs(&g, &g_val)
        .subs(&t, &symplex::default_context().int(1))
        .eval_f64()
        .unwrap();
    println!("    y(t=1) ≈ {:.4} m (f64)", y_f64);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Section 3: Work-Energy Theorem
    //   F = ma (named arithmetic, compile-time checked)
    //   W = ∫F dx → Energy
    //   KE = ½mv² using expr!
    //   Conceptually: W = ΔKE
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Work-Energy Theorem ──");

    // Named typed variables — compile-time dimension checking
    let mass = Mass::symbol("m");
    let accel = Acceleration::symbol("a");

    // F = ma — Mass × Acceleration → Force (compile-time verified!)
    let force = symplex::dim!(Force: mass * accel);
    println!("  F = m·a = {}", force);

    // Work: W = ∫F dx → Energy (typed integration!)
    let x_var = Length::symbol("x");
    let work: Energy = force.integrate_wrt(&x_var);
    println!("  W = ∫F dx = {}", work);

    // Kinetic energy: KE = ½mv² using expr!
    let __ctx = symplex::default_context();
    symplex::syms!(__ctx; m, v);
    let ke = Energy::from_ex(expr!(1/2 * m * v^2));
    println!("  KE = ½mv² = {}", ke);

    // Differentiate KE w.r.t. velocity → Momentum (p = mv)
    let v_var = Velocity::symbol("v");
    let momentum: Momentum = ke.diff_wrt(&v_var);
    println!("  dKE/dv = p = {}", momentum);

    // Differentiate momentum w.r.t. time → Force (Newton's 2nd law)
    let t_typed = Time::symbol("t");
    let force_from_p: Force = momentum.diff_wrt(&t_typed);
    println!("  dp/dt = F = {}", force_from_p);

    // Numerical: m=2kg moving at v=5m/s → KE = 25 J
    let ke_num = ke
        .subs(&m, &symplex::default_context().int(2))
        .subs(&v, &symplex::default_context().int(5))
        .eval();
    println!("\n  KE(m=2, v=5) = {} (should be 25 J)", ke_num);

    // Work = F·d = ma·d. With m=2, a=3, d=10 → W = 60 J
    let work_num = work
        .subs(&mass, &symplex::default_context().int(2))
        .subs(&accel, &symplex::default_context().int(3))
        .subs(&x_var, &symplex::default_context().int(10))
        .eval();
    println!("  W(m=2, a=3, x=10) = {} (should be 60 J)", work_num);

    // Conceptual demonstration: W = ΔKE
    // If a 2kg object accelerates from 0 to v under force F=ma,
    // after distance d: v² = 2·a·d → KE = ½·m·2·a·d = m·a·d = W ✓
    println!("  ✓ Work-Energy Theorem: W = ΔKE (dimensions verified!)");

    println!("\n✓ All done!");
}
