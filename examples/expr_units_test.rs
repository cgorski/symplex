//! Comprehensive demonstration of symplex units + expr! + calculus workflows.
//!
//! Three main patterns:
//!   1. `expr!` + `from_ex()` — most ergonomic for complex formulas
//!   2. Named type arithmetic — compile-time dimension checking for simple products
//!   3. `diff_wrt` / `integrate_wrt` — typed calculus via DiffWrt/IntWrt traits
//!
//! Run with: cargo run --example expr_units_test

use symplex::prelude::*;
use symplex::units::*;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("   Symplex: expr! + Units + Calculus — Full Workflow Demo");
    println!("═══════════════════════════════════════════════════════════════\n");

    pattern_1_expr_and_from_ex();
    pattern_2_named_arithmetic();
    pattern_3_typed_calculus_diff_wrt();
    pattern_4_kinematics_chain();
    pattern_5_electrical_power();
    pattern_6_pendulum_lagrangian();
    pattern_7_spring_mass_damper();
    pattern_8_unit_conversions();
    pattern_9_compile_time_assertions();

    println!("═══════════════════════════════════════════════════════════════");
    println!("   Recommended Workflow Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  a) expr! + from_ex()        — MOST ERGONOMIC");
    println!("     Build complex formulas with natural math syntax.");
    println!("     Example: Energy::from_ex(expr!(1/2 * m * v^2))");
    println!();
    println!("  b) Named type arithmetic    — COMPILE-TIME CHECKED");
    println!("     Simple products where the mul/div table has a result.");
    println!("     Example: let f: Force = &m * &a;");
    println!();
    println!("  c) diff_wrt / integrate_wrt — UNIQUE FEATURE");
    println!("     Typed calculus: the compiler verifies physical laws.");
    println!("     Example: let v: Velocity = position.diff_wrt(&t_var);");
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("   ✓ All 9 patterns demonstrated successfully!");
    println!("═══════════════════════════════════════════════════════════════");
}

// ─────────────────────────────────────────────────────────────────────────
// Pattern 1: Build with expr!, type with from_ex()
//   The most ergonomic approach for complex symbolic expressions.
// ─────────────────────────────────────────────────────────────────────────
fn pattern_1_expr_and_from_ex() {
    println!("── Pattern 1: expr! + from_ex() ──");

    // Declare raw Ex variables for use inside expr!
    symplex::vars!(m, v, a, t, g, k, x);

    // Build complex expressions ergonomically with expr!, wrap with from_ex()
    // from_ex() accepts both Ex and &Ex — no .clone() needed!
    let ke = Energy::from_ex(expr!(1/2 * m * v^2));
    println!("  KE = ½mv² = {}", ke);

    let pe = Energy::from_ex(expr!(m * g * x));
    println!("  PE = mgx  = {}", pe);

    // Constant gravity as an Acceleration
    let grav = Acceleration::from_ex(expr!(g));
    println!("  g = {}", grav);

    // Force = -dPE/dx (Hooke's law from spring PE)
    let spring_pe = Energy::from_ex(expr!(1/2 * k * x^2));
    let spring_force = Force::from_ex(-spring_pe.diff(&x));
    println!("  F = -d(½kx²)/dx = {}", spring_force);

    // Substitute numerical values — dimension preserved!
    let ke_val = ke.clone()
        .subs(&m, &symplex::int(2))
        .subs(&v, &symplex::int(3))
        .eval();
    println!("  KE(m=2, v=3) = {}", ke_val);

    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Pattern 2: Named type arithmetic (compile-time dimension checking)
//   Best for simple products/quotients where the multiplication table
//   has a named result type.
// ─────────────────────────────────────────────────────────────────────────
fn pattern_2_named_arithmetic() {
    println!("── Pattern 2: Named Type Arithmetic ──");

    let mass = Mass::symbol("m");
    let accel = Acceleration::symbol("a");

    // Mass × Acceleration → Force (compile-time checked!)
    let force: Force = &mass * &accel;
    println!("  F = m·a = {}", force);

    // Force + Force → Force (same-type addition)
    let gravity = Force::symbol("F_g");
    let total: Force = &force + &gravity;
    println!("  F_total = F + F_g = {}", total);

    // Scalar multiplication preserves dimension
    let doubled: Force = &force * 2;
    println!("  2F = {}", doubled);

    // Division: Force / Mass → Acceleration
    let a_back: Acceleration = &force / &mass;
    println!("  F/m = {}", a_back);

    // Negation preserves dimension
    let reaction: Force = -&force;
    println!("  -F = {}", reaction);

    // COMPILE-TIME ERROR if uncommented:
    // let bad = &mass + &force;  // ERROR: expected Mass, found Force

    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Pattern 3: Typed calculus with DiffWrt / IntWrt
//   The compiler verifies that differentiating Velocity w.r.t. Time
//   produces Acceleration. No manual type wrapping needed.
// ─────────────────────────────────────────────────────────────────────────
fn pattern_3_typed_calculus_diff_wrt() {
    println!("── Pattern 3: Typed Calculus (DiffWrt / IntWrt) ──");

    symplex::vars!(a, t);

    // Create typed variables
    let t_var = Time::symbol("t");

    // Build a position expression: x(t) = ½at²
    let position = Length::from_ex(expr!(1/2 * a * t^2));
    println!("  x(t) = {}", position);

    // Typed differentiation: d(Length)/d(Time) → Velocity
    let velocity: Velocity = position.diff_wrt(&t_var);
    println!("  v(t) = dx/dt = {}", velocity);

    // Again: d(Velocity)/d(Time) → Acceleration
    let acceleration: Acceleration = velocity.diff_wrt(&t_var);
    println!("  a(t) = dv/dt = {}", acceleration);

    // Typed integration: ∫ Acceleration dt → Velocity
    let v_integrated: Velocity = acceleration.integrate_wrt(&t_var);
    println!("  ∫a dt = {}", v_integrated);

    // ∫ Velocity dt → Length
    let x_integrated: Length = velocity.integrate_wrt(&t_var);
    println!("  ∫v dt = {}", x_integrated);

    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Pattern 4: Full kinematics chain — position → velocity → acceleration
// ─────────────────────────────────────────────────────────────────────────
fn pattern_4_kinematics_chain() {
    println!("── Pattern 4: Kinematics Chain ──");

    symplex::vars!(g, t, v0, x0);

    // Free-fall: x(t) = x₀ + v₀t + ½gt²
    let position = Length::from_ex(expr!(x0 + v0 * t + 1/2 * g * t^2));
    println!("  x(t) = {}", position);

    // v = dx/dt  (use untyped diff + from_ex wrapping)
    let velocity = Velocity::from_ex(position.diff(&t));
    println!("  v(t) = dx/dt = {}", velocity);

    // a = dv/dt
    let acceleration = Acceleration::from_ex(velocity.diff(&t));
    println!("  a(t) = dv/dt = {}", acceleration);

    // Substitute: g=9.81, t=2, v0=5, x0=0
    let x_num = position.clone()
        .subs(&g, &symplex::rational(981, 100))
        .subs(&t, &symplex::int(2))
        .subs(&v0, &symplex::int(5))
        .subs(&x0, &symplex::int(0))
        .eval();
    println!("  x(g=9.81, t=2, v0=5, x0=0) = {}", x_num);

    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Pattern 5: Ohm's Law, Power, and dP/dI
// ─────────────────────────────────────────────────────────────────────────
fn pattern_5_electrical_power() {
    println!("── Pattern 5: Electrical — V=IR, P=IV, dP/dI ──");

    // Named typed variables
    let i_cur = Current::symbol("I");
    let r = Resistance::symbol("R");

    // V = IR (named Mul: Current × Resistance → Voltage)
    let v: Voltage = &i_cur * &r;
    println!("  V = IR = {}", v);

    // P = IV (named Mul: Current × Voltage → Power)
    let p: Power = &i_cur * &v;
    println!("  P = IV = {}", p);

    // Expand to see I²R form
    let p_expanded = p.clone().expand();
    println!("  P expanded = {}", p_expanded);

    // dP/dI using typed DiffWrt: Power / Current → Voltage
    let dp_di: Voltage = p.diff_wrt(&i_cur);
    println!("  dP/dI = {} (should be 2IR = 2V)", dp_di);

    // Numerical check: I=3A, R=10Ω → P=90W, dP/dI=60V
    let i_ex = i_cur.inner();
    let r_ex = r.inner();
    let p_val = p.eval_f64_with(&[(i_ex, 3), (r_ex, 10)]);
    println!("  P(I=3, R=10) = {:?}", p_val);

    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Pattern 6: Pendulum Lagrangian — the full physics workflow
//   T = ½mL²θ̇²   V = mgL(1 - cos θ)   L = T - V
//   ∂L/∂θ̇ → angular momentum   ∂L/∂θ → torque
// ─────────────────────────────────────────────────────────────────────────
fn pattern_6_pendulum_lagrangian() {
    println!("── Pattern 6: Pendulum Lagrangian ──");

    // Raw vars for expr! — most ergonomic for complex formulas
    symplex::vars!(m, l, g, theta, theta_dot);

    // Typed variables for DiffWrt
    let theta_var = Angle::symbol("theta");
    let theta_dot_var = AngularVelocity::symbol("theta_dot");

    // ── Build energies with expr! ──
    let ke = Energy::from_ex(expr!(1/2 * m * l^2 * theta_dot^2));
    let pe = Energy::from_ex(expr!(m * g * l * (1 - cos(theta))));
    println!("  T = {}", ke);
    println!("  V = {}", pe);

    // ── Lagrangian: Energy - Energy = Energy (dimension checked!) ──
    let lagrangian: Energy = &ke - &pe;
    println!("  L = T - V = {}", lagrangian);

    // ── ∂L/∂θ̇ = angular momentum (typed DiffWrt!) ──
    let dl_dthetadot: AngularMomentum = lagrangian.diff_wrt(&theta_dot_var);
    println!("  ∂L/∂θ̇ = {}", dl_dthetadot);

    // ── ∂L/∂θ = torque (typed DiffWrt!) ──
    let dl_dtheta: Torque = lagrangian.diff_wrt(&theta_var);
    println!("  ∂L/∂θ = {}", dl_dtheta);

    // ── Simplify preserves dimension ──
    let dl_dtheta_simplified = dl_dtheta.simplify();
    println!("  ∂L/∂θ simplified = {}", dl_dtheta_simplified);

    // ── Substitute and evaluate ──
    let torque_at = dl_dtheta_simplified
        .subs(&m, &symplex::int(1))
        .subs(&g, &symplex::rational(981, 100))
        .subs(&l, &symplex::rational(1, 2))
        .subs(&theta, &symplex::rational(1, 10))  // θ = 0.1 rad
        .eval();
    println!("  τ(m=1, g=9.81, l=0.5, θ=0.1) = {}", torque_at);

    // ── LaTeX output works on typed quantities ──
    let latex = ke.to_latex();
    println!("  KE in LaTeX: {}", latex);

    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Pattern 7: Spring-Mass-Damper — F = -kx - cv + F_ext
// ─────────────────────────────────────────────────────────────────────────
fn pattern_7_spring_mass_damper() {
    println!("── Pattern 7: Spring-Mass-Damper ──");

    // Use named types for the force equation
    let k = Stiffness::symbol("k");
    let x = Length::symbol("x");
    let c = Damping::symbol("c");
    let v = Velocity::symbol("v");

    // Named Mul: Stiffness × Length → Force, Damping × Velocity → Force
    let f_spring: Force = -(&k * &x);
    let f_damper: Force = -(&c * &v);
    let f_ext = Force::symbol("F_ext");

    // Force + Force + Force → Force (same-type addition)
    let f_total: Force = &(&f_spring + &f_damper) + &f_ext;
    println!("  F = -kx - cv + F_ext = {}", f_total);

    // Newton's law: a = F/m
    let mass = Mass::symbol("m");
    let accel: Acceleration = &f_total / &mass;
    println!("  a = F/m = {}", accel);

    // Expand to see full form
    let accel_expanded = accel.expand();
    println!("  a expanded = {}", accel_expanded);

    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Pattern 8: Unit conversions — constructors that normalize to SI
// ─────────────────────────────────────────────────────────────────────────
fn pattern_8_unit_conversions() {
    println!("── Pattern 8: Unit Conversions ──");

    symplex::vars!(val);

    // Length: 5 kilometers → meters
    let five = symplex::int(5);
    let distance = Length::kilometers(&five);
    println!("  5 km = {}", distance.eval());

    // Power: 120 horsepower → watts
    let hp_val = symplex::int(120);
    let power = Power::horsepower(&hp_val);
    println!("  120 hp = {}", power.eval());

    // Temperature: 100°C → kelvin
    let boiling = symplex::int(100);
    let temp = Temperature::from_celsius(&boiling);
    println!("  100°C = {}", temp.eval());

    // Angle: 90 degrees → radians
    let right_angle = symplex::int(90);
    let angle = Angle::degrees(&right_angle);
    println!("  90° = {} rad", angle);

    // Pressure: 1 atmosphere → pascals
    let one = symplex::int(1);
    let atm = Pressure::atmospheres(&one);
    println!("  1 atm = {}", atm.eval());

    // Frequency: 3600 RPM → hertz
    let rpm_val = symplex::int(3600);
    let freq = Frequency::rpm(&rpm_val);
    println!("  3600 RPM = {}", freq.eval());

    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Pattern 9: Compile-time formula verification
// ─────────────────────────────────────────────────────────────────────────

// These are verified at COMPILE TIME — wrong formulas won't compile!
symplex::const_assert_dim!(
    ConstDim::MASS.mul(ConstDim::ACCELERATION),
    ConstDim::FORCE,
    "F = ma: Mass × Acceleration must equal Force"
);

symplex::const_assert_dim!(
    ConstDim::FORCE.mul(ConstDim::LENGTH),
    ConstDim::ENERGY,
    "W = Fd: Force × Length must equal Energy"
);

symplex::const_assert_dim!(
    ConstDim::ENERGY.div(ConstDim::TIME),
    ConstDim::POWER,
    "P = E/t: Energy / Time must equal Power"
);

symplex::const_assert_dim!(
    ConstDim::CURRENT.mul(ConstDim::RESISTANCE),
    ConstDim::VOLTAGE,
    "V = IR: Current × Resistance must equal Voltage"
);

symplex::const_assert_dim!(
    ConstDim::LENGTH.div(ConstDim::TIME),
    ConstDim::VELOCITY,
    "v = dx/dt: Length / Time must equal Velocity"
);

fn pattern_9_compile_time_assertions() {
    println!("── Pattern 9: Compile-Time Assertions ──");
    println!("  ✓ F = ma     (verified at compile time)");
    println!("  ✓ W = Fd     (verified at compile time)");
    println!("  ✓ P = E/t    (verified at compile time)");
    println!("  ✓ V = IR     (verified at compile time)");
    println!("  ✓ v = dx/dt  (verified at compile time)");

    // Runtime assert_dim! checkpoint
    let m = Mass::symbol("m");
    let a = Acceleration::symbol("a");
    let f = symplex::assert_dim!(&m * &a, Force);
    println!("  ✓ assert_dim!(m*a, Force) = {}", f);

    println!();
}
