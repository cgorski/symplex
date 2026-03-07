//! Compile-time dimensional analysis for physics — a symplex demo.
//!
//! Every multiplication, division, and addition in this file is verified
//! at compile time by Rust's type system.  Try changing a `Force` annotation
//! to `Velocity` and watch the compiler refuse.
//!
//! Patterns demonstrated:
//!   1. `expr!` + `from_ex()` — ergonomic formula building
//!   2. Named type arithmetic — compile-time dimension checking
//!   3. `diff_wrt` / `integrate_wrt` — typed calculus via DiffWrt/IntWrt traits
//!
//! Run with: cargo run --example units_physics

use symplex::prelude::*;
use symplex::units::*;

// ═══════════════════════════════════════════════════════════════════════════
// Compile-time formula verification (checked before main() even runs)
// ═══════════════════════════════════════════════════════════════════════════

// F = ma
symplex::const_assert_dim!(
    ConstDim::MASS.mul(ConstDim::ACCELERATION),
    ConstDim::FORCE,
    "F = ma: Mass * Acceleration must equal Force"
);

// E = F·d
symplex::const_assert_dim!(
    ConstDim::FORCE.mul(ConstDim::LENGTH),
    ConstDim::ENERGY,
    "E = Fd: Force * Length must equal Energy"
);

// P = E / t
symplex::const_assert_dim!(
    ConstDim::ENERGY.div(ConstDim::TIME),
    ConstDim::POWER,
    "P = E/t: Energy / Time must equal Power"
);

// V = IR
symplex::const_assert_dim!(
    ConstDim::CURRENT.mul(ConstDim::RESISTANCE),
    ConstDim::VOLTAGE,
    "V = IR: Current * Resistance must equal Voltage"
);

// P = IV
symplex::const_assert_dim!(
    ConstDim::CURRENT.mul(ConstDim::VOLTAGE),
    ConstDim::POWER,
    "P = IV: Current * Voltage must equal Power"
);

// τ = Iα  (moment of inertia × angular acceleration = torque)
symplex::const_assert_dim!(
    ConstDim::MOMENT_OF_INERTIA.mul(ConstDim::ANGULAR_ACCELERATION),
    ConstDim::TORQUE,
    "tau = I*alpha: MomentOfInertia * AngularAcceleration must equal Torque"
);

fn main() {
    println!("=== Symplex: Compile-Time Dimensional Analysis ===\n");

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 1. Newton's Second Law: F = ma
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("── Newton's Second Law: F = ma ──");

    let m = Mass::symbol("m");
    let a = Acceleration::symbol("a");

    // The type annotation `: Force` is a compile-time assertion.
    // If Mass × Acceleration didn't produce Force, this wouldn't compile.
    let f = symplex::dim!(Force: m * a);
    println!("  F = m·a = {}", f);

    // Substitute numerical values: m = 10 kg, a = 9.81 m/s²
    let f_num = f.clone()
        .subs(&m, &symplex::rational(10, 1))
        .subs(&a, &symplex::rational(981, 100))
        .eval();
    println!("  F(m=10, a=9.81) = {}", f_num);

    // Work done: W = F·d
    let d = Length::symbol("d");
    let w = symplex::dim!(Energy: f * d);
    println!("  W = F·d = {}", w);

    // Power: P = F·v
    let v = Velocity::symbol("v");
    let p = symplex::dim!(Power: f * v);
    println!("  P = F·v = {}", p);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 2. Electrical: Ohm's Law & Power
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Ohm's Law: V = IR, P = IV ──");

    let i = Current::symbol("I");
    let r = Resistance::symbol("R");

    // Ohm's law — the compiler verifies Current × Resistance = Voltage
    let volt = symplex::dim!(Voltage: i * r);
    println!("  V = I·R = {}", volt);

    // Electrical power — Current × Voltage = Power
    let p_elec = symplex::dim!(Power: i * volt);
    println!("  P = I·V = {}", p_elec);

    // Expand P = I·(I·R) to see P = I²·R
    let p_expanded = p_elec.clone().expand();
    println!("  P expanded = {}", p_expanded);

    // Numerical: I = 3 A, R = 47 Ω → V = 141 V, P = 423 W
    let v_num = volt.clone()
        .subs(&i, &symplex::rational(3, 1))
        .subs(&r, &symplex::rational(47, 1))
        .eval();
    println!("  V(I=3, R=47) = {}", v_num);

    let p_num = p_elec
        .subs(&i, &symplex::rational(3, 1))
        .subs(&r, &symplex::rational(47, 1))
        .eval();
    println!("  P(I=3, R=47) = {}", p_num);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 3. Pendulum: Potential Energy with Trig
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Pendulum: V = mgl(1 - cos θ) ──");

    let m_pend = Mass::symbol("m");
    let g = Acceleration::symbol("g");
    let l = Length::symbol("l");
    let theta = Angle::symbol("θ");

    // Build up step by step — each intermediate has a verified type:
    //   Mass × Acceleration = Force
    //   Force × Length = Energy
    let weight = symplex::dim!(Force: m_pend * g);
    let mgl = symplex::dim!(Energy: weight * l);
    println!("  m·g·l = {}", mgl);

    // Angle::cos() returns Dimensionless — enforced by the type system.
    // sin/cos/tan are *only* callable on Angle, not on arbitrary quantities.
    let cos_theta: Dimensionless = theta.cos();
    let one_minus_cos: Dimensionless = Dimensionless::constant(1) - cos_theta;

    // Energy × Dimensionless = Energy (dimensionless scaling preserves units)
    let pe = symplex::dim!(Energy: mgl * one_minus_cos);
    println!("  V = m·g·l·(1 - cos θ) = {}", pe);

    // Restoring torque: τ = -mgl·sin(θ)
    // Energy and Torque share the same dimension vector (L²·M·T⁻²),
    // so we convert explicitly with Torque::from_energy().
    let sin_theta: Dimensionless = Angle::symbol("θ").sin();
    // Reuse the mgl Energy we already built, scale by sin(θ).
    // Energy × Dimensionless = Energy (dimensionless scaling).
    let torque_magnitude = symplex::dim!(Energy: mgl * sin_theta);
    let tau: Torque = Torque::from_energy(-torque_magnitude);
    println!("  τ = -m·g·l·sin(θ) = {}", tau);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 4. Spring-Mass-Damper System
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Spring-Mass-Damper: F = -kx - cv + F_ext ──");

    let k = Stiffness::symbol("k");
    let x = Length::symbol("x");
    let c = Damping::symbol("c");
    let v_smd = Velocity::symbol("v");
    let f_ext = Force::symbol("F_ext");

    // Spring force: Stiffness × Length = Force
    let f_spring = symplex::dim!(Force: -(k * x));
    println!("  F_spring = -kx = {}", f_spring);

    // Damping force: Damping × Velocity = Force
    let f_damp = symplex::dim!(Force: -(c * v_smd));
    println!("  F_damp  = -cv = {}", f_damp);

    // Total force — addition is type-safe: Force + Force = Force
    let f_total: Force = &f_spring + &f_damp + f_ext.clone();
    println!("  F_total = {}", f_total);

    // Acceleration from Newton's second law: Force / Mass = Acceleration
    let m_smd = Mass::symbol("m");
    let a_smd = symplex::dim!(Acceleration: f_total / m_smd);
    println!("  a = F/m = {}", a_smd);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 5. DC Motor (Steady-State): V = R·I + Ke·ω
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── DC Motor Steady-State: V = R·I + Ke·ω ──");

    let r_motor = Resistance::symbol("R");
    let i_motor = Current::symbol("I");
    let omega = AngularVelocity::symbol("ω");

    // Back-EMF constant Ke has units of Wb (V·s/rad ≡ magnetic flux)
    // MagneticFlux × AngularVelocity = Voltage
    let ke = MagneticFlux::symbol("Ke");

    // Resistive voltage drop: Resistance × Current = Voltage
    let v_resistive = symplex::dim!(Voltage: r_motor * i_motor);
    println!("  V_R  = R·I  = {}", v_resistive);

    // Back-EMF: MagneticFlux × AngularVelocity = Voltage
    let v_emf = symplex::dim!(Voltage: ke * omega);
    println!("  V_emf = Ke·ω = {}", v_emf);

    // Total supply voltage — Voltage + Voltage = Voltage
    let v_supply: Voltage = &v_resistive + &v_emf;
    println!("  V = R·I + Ke·ω = {}", v_supply);

    // Electrical power into the motor
    let p_motor = symplex::dim!(Power: i_motor * v_supply);
    println!("  P_in = I·V = {}", p_motor);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 6. Unit Conversions
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Unit Conversions ──");

    // All quantities store SI internally; constructors convert automatically.
    let distance = Length::kilometers(&symplex::rational(5, 1));
    println!("  5 km = {}", distance);

    let engine = Power::horsepower(&symplex::rational(300, 1));
    println!("  300 hp = {}", engine);

    let boiling = Temperature::from_celsius(&symplex::rational(100, 1));
    println!("  100 °C = {}", boiling);

    let body_temp = Temperature::from_fahrenheit(&symplex::rational(986, 10));
    println!("  98.6 °F = {}", body_temp);

    let highway = Velocity::kilometers_per_hour(&symplex::rational(120, 1));
    println!("  120 km/h = {}", highway);

    let one_g = Acceleration::standard_gravity(&symplex::int(1));
    println!("  1 g = {}", one_g);

    let atm = Pressure::atmospheres(&symplex::rational(1, 1));
    println!("  1 atm = {}", atm);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 7. Typed Calculus with DiffWrt / IntWrt
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Typed Calculus (DiffWrt / IntWrt) ──");

    // DiffWrt and IntWrt traits let you differentiate and integrate
    // named types directly — the compiler verifies the physical law.

    symplex::vars!(a, t);

    let t_var = Time::symbol("t");

    // Position x(t) = ½at²  (built with expr!, typed with from_ex)
    let position = Length::from_ex(expr!(1/2 * a * t^2));
    println!("  x(t) = {}", position);

    // d(Length)/d(Time) → Velocity  (compiler-verified!)
    let velocity: Velocity = position.diff_wrt(&t_var);
    println!("  v(t) = dx/dt = {}", velocity);

    // d(Velocity)/d(Time) → Acceleration  (compiler-verified!)
    let acceleration: Acceleration = velocity.diff_wrt(&t_var);
    println!("  a(t) = dv/dt = {}", acceleration);

    // ∫ Acceleration dt → Velocity  (IntWrt: reverse of differentiation)
    let v_back: Velocity = acceleration.integrate_wrt(&t_var);
    println!("  ∫ a dt = {} (Velocity)", v_back);

    // ∫ Velocity dt → Length
    let x_back: Length = velocity.integrate_wrt(&t_var);
    println!("  ∫ v dt = {} (Length)", x_back);

    // ── More DiffWrt examples ──

    // Energy / Time → Power
    let energy = Energy::from_ex(expr!(1/2 * a * t^2));
    let power: Power = energy.diff_wrt(&t_var);
    println!("  dE/dt = {} (Power)", power);

    // Momentum / Time → Force (Newton's second law: F = dp/dt)
    let momentum = Momentum::from_ex(expr!(a * t));
    let force_from_p: Force = momentum.diff_wrt(&t_var);
    println!("  dp/dt = {} (Force)", force_from_p);

    // Charge / Time → Current (I = dQ/dt)
    let charge = Charge::from_ex(expr!(a * t));
    let current_from_q: Current = charge.diff_wrt(&t_var);
    println!("  dQ/dt = {} (Current)", current_from_q);

    // MagneticFlux / Time → Voltage (Faraday's law: V = dΦ/dt)
    let flux = MagneticFlux::from_ex(expr!(a * t));
    let emf: Voltage = flux.diff_wrt(&t_var);
    println!("  dΦ/dt = {} (Voltage — Faraday's law)", emf);

    // Energy / Length → Force (F = -dU/dx)
    symplex::vars!(k, x);
    let x_var = Length::symbol("x");
    let spring_pe = Energy::from_ex(expr!(1/2 * k * x^2));
    let spring_force: Force = spring_pe.diff_wrt(&x_var);
    println!("  dU/dx = {} (Force from spring PE)", spring_force);

    // Power / Current → Voltage (dP/dI)
    symplex::vars!(i_p, r_p);
    let i_var = Current::symbol("i_p");
    let power_expr = Power::from_ex(expr!(i_p^2 * r_p));
    let dp_di: Voltage = power_expr.diff_wrt(&i_var);
    println!("  dP/dI = {} (Voltage)", dp_di);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 8. Dimension-Checked Assertions (runtime demo of compile-time safety)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── assert_dim! Macro ──");

    // assert_dim! is a compile-time check disguised as an expression.
    // If the dimension is wrong, the code won't compile at all.
    let m_check = Mass::symbol("m");
    let a_check = Acceleration::symbol("a");
    let f_check = symplex::dim!(Force: m_check * a_check);
    println!("  dim!(Force: m*a) ✓ = {}", f_check);

    let v_check = Velocity::symbol("v");
    let t_check = Time::symbol("t");
    let x_check = symplex::dim!(Length: v_check * t_check);
    println!("  dim!(Length: v*t) ✓ = {}", x_check);

    // Momentum: Mass × Velocity
    let p_check = symplex::dim!(Momentum: m_check * v_check);
    println!("  dim!(Momentum: m*v) ✓ = {}", p_check);

    // Momentum: Mass × Velocity = Momentum (verified at compile time)
    let impulse = symplex::dim!(Momentum: m_check * v_check);
    println!("  dim!(Momentum: m*v) ✓ = {}", impulse);

    // Energy: Force × Length
    let l_check = Length::symbol("d");
    let w_check = symplex::dim!(Energy: f_check * l_check);
    println!("  dim!(Energy: F*d) ✓ = {}", w_check);

    // Power: Energy / Time
    let pw_check = symplex::dim!(Power: w_check / t_check);
    println!("  dim!(Power: E/t) ✓ = {}", pw_check);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Compile-Time Formula Verification (const_assert_dim!) ──");
    println!("  ✓ F = ma   (Mass × Acceleration = Force)");
    println!("  ✓ E = Fd   (Force × Length = Energy)");
    println!("  ✓ P = E/t  (Energy / Time = Power)");
    println!("  ✓ V = IR   (Current × Resistance = Voltage)");
    println!("  ✓ P = IV   (Current × Voltage = Power)");
    println!("  ✓ τ = Iα   (MomentOfInertia × AngularAcceleration = Torque)");
    println!("  (All verified at compile time — see top of file)");

    println!("\n── Physical Constants ──");
    {
        use symplex::units::constants;
        let c = constants::speed_of_light();
        let m = Mass::symbol("m");
        let energy = symplex::dim!(Energy: m * c * c);
        println!("  E = mc² = {}", energy);
        println!("  (Displays symbolically — 'c' not '299792458')");
        println!("  E(m=1kg) = {:.3e} J", energy.subs(&m, &symplex::int(1)).eval_f64().unwrap());
    }

    println!("\n✓ All dimensional checks passed!");
}
