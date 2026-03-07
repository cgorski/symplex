//! Electrical circuits with compile-time dimensional analysis.
//!
//! Demonstrates: Ohm's law, RC circuit, DC motor equation, power analysis.
//!
//! Run with: cargo run --example units_electrical

use symplex::prelude::*;
use symplex::units::*;

#[allow(non_snake_case)]
fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("   Symplex: Electrical Circuits — Dimensional Analysis");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Section 1: Ohm's Law & Power
    //   V = IR, P = IV, P = I²R, dP/dI = 2IR
    //   Named type arithmetic — every product is compile-time checked.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("── Ohm's Law & Power ──");

    // Named typed variables — the compiler tracks their dimensions
    let i = Current::symbol("I");
    let r = Resistance::symbol("R");

    // V = IR — Current × Resistance → Voltage (compile-time checked!)
    let v: Voltage = &i * &r;
    println!("  V = I·R = {}", v);

    // P = IV — Current × Voltage → Power (compile-time checked!)
    let p: Power = &i * &v;
    println!("  P = I·V = {}", p);

    // Expand P = I·(I·R) to see the I²R form
    let p_expanded = p.clone().expand();
    println!("  P expanded = {}", p_expanded);

    // dP/dI: typed differentiation — Power / Current → Voltage
    // The compiler verifies that differentiating Power w.r.t. Current
    // produces Voltage (since [W]/[A] = [V]).
    let dp_di: Voltage = p.diff_wrt(&i);
    println!("  dP/dI = {} (should be 2·I·R)", dp_di);

    // Numerical evaluation: I = 3 A, R = 47 Ω
    let v_num = v.clone()
        .subs(&i, &symplex::int(3))
        .subs(&r, &symplex::int(47))
        .eval();
    println!("\n  Numerical (I=3 A, R=47 Ω):");
    println!("    V = {}", v_num);

    let p_num = p.clone()
        .subs(&i, &symplex::int(3))
        .subs(&r, &symplex::int(47))
        .eval();
    println!("    P = {}", p_num);

    let dp_di_num = dp_di
        .subs(&i, &symplex::int(3))
        .subs(&r, &symplex::int(47))
        .eval();
    println!("    dP/dI = {}", dp_di_num);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Section 2: RC Circuit Time Constant
    //   τ = R·C
    //   V_C(t) = V₀·(1 − e^(−t/(R·C)))
    //   Built with expr! for the complex exponential formula.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── RC Circuit Time Constant ──");

    // Raw variables for expr! — most ergonomic for the exponential formula
    symplex::vars!(R, C, V0, t);

    // Time constant τ = R·C (has dimension of Time)
    let tau = Time::from_ex(expr!(R * C));
    println!("  τ = R·C = {}", tau);

    // Capacitor voltage during charging:
    //   V_C(t) = V₀·(1 − e^(−t/(R·C)))
    let v_cap = Voltage::from_ex(expr!(V0 * (1 - exp(-t / (R * C)))));
    println!("  V_C(t) = {}", v_cap);

    // Charging current: I(t) = dV_C/dt · C = (V₀/R)·e^(−t/(R·C))
    // We differentiate V_C w.r.t. t (raw diff) and wrap as the rate
    let dv_dt = v_cap.diff(&t);
    println!("  dV_C/dt = {}", dv_dt);

    // Numerical: R = 1 kΩ = 1000 Ω, C = 100 µF = 1/10000 F,
    //            V₀ = 5 V, t = 0.1 s
    // τ = 1000 × 0.0001 = 0.1 s, so at t = τ we expect ≈ 63.2% of V₀
    let tau_num = tau
        .subs(&R, &symplex::int(1000))
        .subs(&C, &symplex::rational(1, 10000))
        .eval();
    println!("\n  Numerical (R=1kΩ, C=100µF, V₀=5V):");
    println!("    τ = {}", tau_num);

    let v_cap_at_tau = v_cap.clone()
        .subs(&R, &symplex::int(1000))
        .subs(&C, &symplex::rational(1, 10000))
        .subs(&V0, &symplex::int(5))
        .subs(&t, &symplex::rational(1, 10))
        .eval();
    println!("    V_C(t=0.1s = τ) = {}", v_cap_at_tau);

    // f64 check: should be ≈ 5·(1 - e⁻¹) ≈ 3.1606
    let v_cap_f64 = v_cap
        .subs(&R, &symplex::int(1000))
        .subs(&C, &symplex::rational(1, 10000))
        .subs(&V0, &symplex::int(5))
        .subs(&t, &symplex::rational(1, 10))
        .eval_f64()
        .unwrap();
    println!("    V_C(t=τ) ≈ {:.4} V (f64, expect ≈3.1606)", v_cap_f64);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Section 3: DC Motor Steady-State
    //   V = R·I + Ke·ω
    //   P_in = I·V, P_mech = Ke·ω·I
    //   Named types enforce dimension correctness at every step.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── DC Motor Steady-State ──");

    // Named typed variables for the motor equation
    let r_motor = Resistance::symbol("R_m");
    let i_motor = Current::symbol("I_m");
    let omega = AngularVelocity::symbol("ω");

    // Back-EMF constant Ke has units of Wb (V·s/rad ≡ magnetic flux)
    // MagneticFlux × AngularVelocity → Voltage (from mul_table)
    let ke = MagneticFlux::symbol("Ke");

    // Resistive voltage drop: Resistance × Current → Voltage
    let v_resistive: Voltage = &r_motor * &i_motor;
    println!("  V_R   = R·I = {}", v_resistive);

    // Back-EMF: MagneticFlux × AngularVelocity → Voltage
    let v_emf: Voltage = &ke * &omega;
    println!("  V_emf = Ke·ω = {}", v_emf);

    // Total supply voltage: Voltage + Voltage → Voltage (same-type addition)
    let v_supply: Voltage = &v_resistive + &v_emf;
    println!("  V_supply = R·I + Ke·ω = {}", v_supply);

    // Input electrical power: Current × Voltage → Power
    let p_in: Power = &i_motor * &v_supply;
    println!("  P_in = I·V = {}", p_in);

    // Mechanical output power: Ke·ω·I
    // (MagneticFlux × AngularVelocity → Voltage, then Voltage × Current → Power)
    let p_mech: Power = &i_motor * &v_emf;
    println!("  P_mech = Ke·ω·I = {}", p_mech);

    // Resistive loss: I²R (Current × Voltage_R → Power)
    let p_loss: Power = &i_motor * &v_resistive;
    println!("  P_loss = I²R = {}", p_loss);

    // Power balance: P_in = P_mech + P_loss (conceptual)
    println!("  ✓ P_in = P_mech + P_loss (dimension-verified)");

    // Numerical motor example:
    //   R_m = 2 Ω, Ke = 0.05 Wb, ω = 100 rad/s, → V_emf = 5 V
    //   V_supply = 12 V → I = (V - Ke·ω)/R = (12 - 5)/2 = 3.5 A
    let ke_ex = ke.inner();
    let omega_ex = omega.inner();
    let r_m_ex = r_motor.inner();
    let i_m_ex = i_motor.inner();

    let v_emf_num = v_emf
        .subs(&ke, &symplex::rational(5, 100))
        .subs(&omega, &symplex::int(100))
        .eval();
    println!("\n  Numerical (R=2Ω, Ke=0.05Wb, ω=100rad/s):");
    println!("    V_emf = {}", v_emf_num);

    let p_mech_f64 = p_mech
        .eval_f64_with(&[
            (i_m_ex, 3),
            (ke_ex, 1),   // approximate Ke=1 for integer check
            (omega_ex, 100),
        ])
        .unwrap();
    println!("    P_mech(I=3, Ke≈1, ω=100) ≈ {:.1} W (f64)", p_mech_f64);

    let p_loss_f64 = p_loss
        .eval_f64_with(&[(i_m_ex, 3), (r_m_ex, 2)])
        .unwrap();
    println!("    P_loss(I=3, R=2) = {:.1} W (f64)", p_loss_f64);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Section 4: Unit Conversions
    //   Demonstrate constructor helpers that normalize to SI base units.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n── Unit Conversions ──");

    // Current: 500 milliamps → amperes
    let small_current = Current::milliamperes(&symplex::int(500));
    println!("  500 mA  = {}", small_current.eval());

    // Resistance: 4.7 kilohms → ohms
    let big_resistor = Resistance::kilohms(&symplex::rational(47, 10));
    println!("  4.7 kΩ  = {}", big_resistor.eval());

    // Power: 1 horsepower → watts
    let one_hp = Power::horsepower(&symplex::int(1));
    println!("  1 hp    = {}", one_hp.eval());

    // Frequency: 3600 RPM → hertz
    let motor_speed = Frequency::rpm(&symplex::int(3600));
    println!("  3600 RPM = {}", motor_speed.eval());

    // Capacitance: 100 µF → farads
    let cap = Capacitance::microfarads(&symplex::int(100));
    println!("  100 µF  = {}", cap.eval());

    // Voltage: 3300 mV → volts
    let logic_level = Voltage::millivolts(&symplex::int(3300));
    println!("  3300 mV = {}", logic_level.eval());

    // Charge: 2000 mAh → coulombs
    let battery = Charge::milliampere_hours(&symplex::int(2000));
    println!("  2000 mAh = {}", battery.eval());

    // Inductance: 10 mH → henrys
    let coil = Inductance::millihenrys(&symplex::int(10));
    println!("  10 mH   = {}", coil.eval());

    println!("\n── Physical Constants in Circuits ──");
    {
        use symplex::units::constants;
        let e_charge = constants::elementary_charge();
        println!("  Elementary charge: e = {}", e_charge);
        println!("  e = {:.10e} C", e_charge.eval_f64().unwrap());

        // Energy of an electron accelerated through 1V:
        // E = eV = 1 eV = 1.602e-19 J
        let one_volt = Voltage::constant(1);
        // charge × voltage = energy (Charge × Voltage = Energy in the mul table)
        let energy: Energy = &e_charge * &one_volt;
        println!("  Energy of 1 eV = {} = {:.6e} J", energy, energy.eval_f64().unwrap());
    }

    println!("\n✓ All done!");
}
