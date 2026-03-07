//! Compile-time dimensional analysis for physics — a symplex demo.
//!
//! Every multiplication, division, and addition in this file is verified
//! at compile time by Rust's type system.  Try changing a `Force` annotation
//! to `Velocity` and watch the compiler refuse.
//!
//! Run with: cargo run --example units_physics

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
    let f: Force = &m * &a;
    println!("  F = m·a = {}", f);

    // Substitute numerical values: m = 10 kg, a = 9.81 m/s²
    let f_num = f.clone()
        .subs(m.inner(), &symplex::rational(10, 1))
        .subs(a.inner(), &symplex::rational(981, 100))
        .eval();
    println!("  F(m=10, a=9.81) = {}", f_num);

    // Work done: W = F·d
    let d = Length::symbol("d");
    let w: Energy = &f * &d;
    println!("  W = F·d = {}", w);

    // Power: P = F·v
    let v = Velocity::symbol("v");
    let p: Power = &f * &v;
    println!("  P = F·v = {}", p);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 2. Electrical: Ohm's Law & Power
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Ohm's Law: V = IR, P = IV ──");

    let i = Current::symbol("I");
    let r = Resistance::symbol("R");

    // Ohm's law — the compiler verifies Current × Resistance = Voltage
    let volt: Voltage = &i * &r;
    println!("  V = I·R = {}", volt);

    // Electrical power — Current × Voltage = Power
    let p_elec: Power = &i * &volt;
    println!("  P = I·V = {}", p_elec);

    // Expand P = I·(I·R) to see P = I²·R
    let p_expanded = p_elec.clone().expand();
    println!("  P expanded = {}", p_expanded);

    // Numerical: I = 3 A, R = 47 Ω → V = 141 V, P = 423 W
    let v_num = volt.clone()
        .subs(i.inner(), &symplex::rational(3, 1))
        .subs(r.inner(), &symplex::rational(47, 1))
        .eval();
    println!("  V(I=3, R=47) = {}", v_num);

    let p_num = p_elec
        .subs(i.inner(), &symplex::rational(3, 1))
        .subs(r.inner(), &symplex::rational(47, 1))
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
    let weight: Force = &m_pend * &g;
    let mgl: Energy = &weight * &l;
    println!("  m·g·l = {}", mgl);

    // Angle::cos() returns Dimensionless — enforced by the type system.
    // sin/cos/tan are *only* callable on Angle, not on arbitrary quantities.
    let cos_theta: Dimensionless = theta.cos();
    let one_minus_cos: Dimensionless = Dimensionless::constant(1) - cos_theta;

    // Energy × Dimensionless = Energy (dimensionless scaling preserves units)
    let pe: Energy = mgl * one_minus_cos;
    println!("  V = m·g·l·(1 - cos θ) = {}", pe);

    // Restoring torque: τ = -mgl·sin(θ)
    // Energy and Torque share the same dimension vector (L²·M·T⁻²),
    // so we convert explicitly with Torque::from_energy().
    let sin_theta: Dimensionless = Angle::symbol("θ").sin();
    // Reuse the mgl Energy we already built, scale by sin(θ).
    // Energy × Dimensionless = Energy (dimensionless scaling).
    let mgl2: Energy = &(&Mass::symbol("m") * &Acceleration::symbol("g"))
        * &Length::symbol("l");
    let torque_magnitude: Energy = mgl2 * sin_theta;
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
    let f_spring: Force = -(&k * &x);
    println!("  F_spring = -kx = {}", f_spring);

    // Damping force: Damping × Velocity = Force
    let f_damp: Force = -(&c * &v_smd);
    println!("  F_damp  = -cv = {}", f_damp);

    // Total force — addition is type-safe: Force + Force = Force
    let f_total: Force = &f_spring + &f_damp + f_ext.clone();
    println!("  F_total = {}", f_total);

    // Acceleration from Newton's second law: Force / Mass = Acceleration
    let m_smd = Mass::symbol("m");
    let a_smd: Acceleration = &f_total / &m_smd;
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
    let v_resistive: Voltage = &r_motor * &i_motor;
    println!("  V_R  = R·I  = {}", v_resistive);

    // Back-EMF: MagneticFlux × AngularVelocity = Voltage
    let v_emf: Voltage = &ke * &omega;
    println!("  V_emf = Ke·ω = {}", v_emf);

    // Total supply voltage — Voltage + Voltage = Voltage
    let v_supply: Voltage = &v_resistive + &v_emf;
    println!("  V = R·I + Ke·ω = {}", v_supply);

    // Electrical power into the motor
    let p_motor: Power = &i_motor * &v_supply;
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
    // 7. Differentiation & Integration with Dimensions
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Dimensional Calculus ──");

    // Build expressions where the differentiation variable actually appears,
    // so we get nonzero symbolic derivatives.

    let t_var = symplex::var("t");
    let v_var = symplex::var("v");
    let a_var = symplex::var("a");

    // Position x(t) = v·t  →  dx/dt = v  (Length / Time = Velocity)
    let x_of_t: Qty<LengthDim> = Qty::from_ex(&v_var * &t_var);
    let t_qty: Qty<TimeDim> = Qty::from_ex(t_var.clone());
    let v_calc: Velocity = diff_qty(&x_of_t, &t_qty).into();
    println!("  x(t) = v·t         → dx/dt = {} (Velocity)", v_calc);

    // Velocity v(t) = a·t  →  dv/dt = a  (Velocity / Time = Acceleration)
    let v_of_t: Qty<VelocityDim> = Qty::from_ex(&a_var * &t_var);
    let a_calc: Acceleration = diff_qty(&v_of_t, &t_qty).into();
    println!("  v(t) = a·t         → dv/dt = {} (Acceleration)", a_calc);

    // ∫ Force dx = Energy  (work done by a constant force)
    let f_q: Qty<ForceDim> = Qty::from_ex(symplex::var("F"));
    let x_q: Qty<LengthDim> = Qty::from_ex(symplex::var("x"));
    let work: Energy = integrate_qty(&f_q, &x_q).into();
    println!("  ∫ F dx             → {} (Energy)", work);

    // ∫ Velocity dt = Length  (displacement from constant velocity)
    let v_int: Qty<VelocityDim> = Qty::from_ex(v_var);
    let t_int: Qty<TimeDim> = Qty::from_ex(t_var);
    let disp: Length = integrate_qty(&v_int, &t_int).into();
    println!("  ∫ v dt             → {} (Length)", disp);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 8. Dimension-Checked Assertions (runtime demo of compile-time safety)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── assert_dim! Macro ──");

    // assert_dim! is a compile-time check disguised as an expression.
    // If the dimension is wrong, the code won't compile at all.
    let m_check = Mass::symbol("m");
    let a_check = Acceleration::symbol("a");
    let f_check = symplex::assert_dim!(&m_check * &a_check, Force);
    println!("  assert_dim!(m*a, Force) ✓ = {}", f_check);

    let v_check = Velocity::symbol("v");
    let t_check = Time::symbol("t");
    let x_check = symplex::assert_dim!(&v_check * &t_check, Length);
    println!("  assert_dim!(v*t, Length) ✓ = {}", x_check);

    // Momentum: Mass × Velocity
    let p_check = symplex::assert_dim!(&m_check * &v_check, Momentum);
    println!("  assert_dim!(m*v, Momentum) ✓ = {}", p_check);

    // Momentum: Mass × Velocity = Momentum (verified at compile time)
    let impulse = symplex::assert_dim!(&m_check * &v_check, Momentum);
    println!("  assert_dim!(m*v, Momentum) ✓ = {}", impulse);

    // Energy: Force × Length
    let l_check = Length::symbol("d");
    let w_check = symplex::assert_dim!(&f_check * &l_check, Energy);
    println!("  assert_dim!(F*d, Energy) ✓ = {}", w_check);

    // Power: Energy / Time
    let pw_check = symplex::assert_dim!(&w_check / &t_check, Power);
    println!("  assert_dim!(E/t, Power) ✓ = {}", pw_check);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Compile-Time Formula Verification (const_assert_dim!) ──");
    println!("  ✓ F = ma   (Mass × Acceleration = Force)");
    println!("  ✓ E = Fd   (Force × Length = Energy)");
    println!("  ✓ P = E/t  (Energy / Time = Power)");
    println!("  ✓ V = IR   (Current × Resistance = Voltage)");
    println!("  ✓ P = IV   (Current × Voltage = Power)");
    println!("  ✓ τ = Iα   (MomentOfInertia × AngularAcceleration = Torque)");
    println!("  (All verified at compile time — see top of file)");

    println!("\n✓ All dimensional checks passed!");
}
