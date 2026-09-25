//! Round 2 domain-specific tests — hunting for bugs in under-tested modules.
//!
//! Covers: units/dimensional analysis, matrix edge cases, quaternions,
//! vector calculus, control systems, robotics, and combinatorics.

use symplex::prelude::*;
use symplex::units::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper utilities
// ═══════════════════════════════════════════════════════════════════════════

fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{msg}: expected {expected}, got {actual} (diff = {})",
        (actual - expected).abs()
    );
}

fn quat_to_f64(q: &symplex::quaternion::Quaternion) -> (f64, f64, f64, f64) {
    (
        q.w.eval().eval_f64().unwrap(),
        q.x.eval().eval_f64().unwrap(),
        q.y.eval().eval_f64().unwrap(),
        q.z.eval().eval_f64().unwrap(),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. UNITS / DIMENSIONAL ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

// ---------------------------------------------------------------------------
// 1a. Compile-time dimension assertions via const_assert_dim!
// ---------------------------------------------------------------------------

// Momentum = Mass × Velocity
symplex::const_assert_dim!(
    ConstDim::MASS.mul(ConstDim::VELOCITY),
    ConstDim::MOMENTUM,
    "p = mv: Mass * Velocity must equal Momentum"
);

// Pressure = Force / Area
symplex::const_assert_dim!(
    ConstDim::FORCE.div(ConstDim::AREA),
    ConstDim::PRESSURE,
    "P = F/A: Force / Area must equal Pressure"
);

// Angular momentum = Moment of inertia × Angular velocity
symplex::const_assert_dim!(
    ConstDim::MOMENT_OF_INERTIA.mul(ConstDim::ANGULAR_VELOCITY),
    ConstDim::ANGULAR_MOMENTUM,
    "L = Iω: MomentOfInertia * AngularVelocity must equal AngularMomentum"
);

// Charge = Current × Time
symplex::const_assert_dim!(
    ConstDim::CURRENT.mul(ConstDim::TIME),
    ConstDim::CHARGE,
    "Q = It: Current * Time must equal Charge"
);

// Stiffness = Force / Length
symplex::const_assert_dim!(
    ConstDim::FORCE.div(ConstDim::LENGTH),
    ConstDim::STIFFNESS,
    "k = F/x: Force / Length must equal Stiffness"
);

// Power = Voltage × Current
symplex::const_assert_dim!(
    ConstDim::VOLTAGE.mul(ConstDim::CURRENT),
    ConstDim::POWER,
    "P = VI: Voltage * Current must equal Power"
);

// Inductance × Current = Magnetic flux (V = L dI/dt → L·I ~ Φ)
symplex::const_assert_dim!(
    ConstDim::INDUCTANCE.mul(ConstDim::CURRENT),
    ConstDim::MAGNETIC_FLUX,
    "Φ = LI: Inductance * Current must equal MagneticFlux"
);

// Capacitance × Voltage = Charge
symplex::const_assert_dim!(
    ConstDim::CAPACITANCE.mul(ConstDim::VOLTAGE),
    ConstDim::CHARGE,
    "Q = CV: Capacitance * Voltage must equal Charge"
);

// Energy / Time = Power
symplex::const_assert_dim!(
    ConstDim::ENERGY.div(ConstDim::TIME),
    ConstDim::POWER,
    "P = E/t: Energy / Time must equal Power"
);

// Resistance = Voltage / Current
symplex::const_assert_dim!(
    ConstDim::VOLTAGE.div(ConstDim::CURRENT),
    ConstDim::RESISTANCE,
    "R = V/I: Voltage / Current must equal Resistance"
);

// Force × Time = Momentum (impulse–momentum theorem)
symplex::const_assert_dim!(
    ConstDim::FORCE.mul(ConstDim::TIME),
    ConstDim::MOMENTUM,
    "J = Ft: Force * Time must equal Momentum"
);

// ---------------------------------------------------------------------------
// 1b. ConstDim pow / roundtrip identities
// ---------------------------------------------------------------------------

symplex::const_assert_dim!(
    ConstDim::LENGTH.pow(2),
    ConstDim::AREA,
    "L^2 must equal Area"
);

symplex::const_assert_dim!(
    ConstDim::LENGTH.pow(3),
    ConstDim::VOLUME,
    "L^3 must equal Volume"
);

// Dimension divided by itself is dimensionless
symplex::const_assert_dim!(
    ConstDim::FORCE.div(ConstDim::FORCE),
    ConstDim::DIMENSIONLESS,
    "F/F must be dimensionless"
);

// ---------------------------------------------------------------------------
// 1c. Unit conversions — Length
// ---------------------------------------------------------------------------

#[test]
fn unit_length_kilometers_to_meters() {
    let ctx = Context::new();
    let val = ctx.int(5);
    let l = Length::kilometers(&val);
    let f = l.eval_f64().unwrap();
    assert_close(f, 5000.0, 1e-10, "5 km = 5000 m");
}

#[test]
fn unit_length_inches_to_meters() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let l = Length::inches(&val);
    let f = l.eval_f64().unwrap();
    assert_close(f, 0.0254, 1e-12, "1 inch = 0.0254 m");
}

#[test]
fn unit_length_feet_to_meters() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let l = Length::feet(&val);
    let f = l.eval_f64().unwrap();
    assert_close(f, 0.3048, 1e-12, "1 foot = 0.3048 m");
}

#[test]
fn unit_length_yards_to_meters() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let l = Length::yards(&val);
    let f = l.eval_f64().unwrap();
    assert_close(f, 0.9144, 1e-12, "1 yard = 0.9144 m");
}

#[test]
fn unit_length_miles_to_meters() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let l = Length::miles(&val);
    let f = l.eval_f64().unwrap();
    assert_close(f, 1609.344, 1e-6, "1 mile = 1609.344 m");
}

#[test]
fn unit_length_nautical_miles_to_meters() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let l = Length::nautical_miles(&val);
    let f = l.eval_f64().unwrap();
    assert_close(f, 1852.0, 1e-10, "1 nmi = 1852 m");
}

#[test]
fn unit_length_millimeters_centimeters_micrometers() {
    let ctx = Context::new();
    let val = ctx.int(1);

    let mm = Length::millimeters(&val);
    assert_close(mm.eval_f64().unwrap(), 0.001, 1e-15, "1 mm");

    let cm = Length::centimeters(&val);
    assert_close(cm.eval_f64().unwrap(), 0.01, 1e-15, "1 cm");

    let um = Length::micrometers(&val);
    assert_close(um.eval_f64().unwrap(), 1e-6, 1e-18, "1 μm");
}

// ---------------------------------------------------------------------------
// 1d. Unit conversions — Mass
// ---------------------------------------------------------------------------

#[test]
fn unit_mass_pounds_to_kilograms() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let m = Mass::pounds(&val);
    let f = m.eval_f64().unwrap();
    assert_close(f, 0.45359237, 1e-10, "1 lb = 0.45359237 kg");
}

#[test]
fn unit_mass_grams_to_kilograms() {
    let ctx = Context::new();
    let val = ctx.int(1000);
    let m = Mass::grams(&val);
    let f = m.eval_f64().unwrap();
    assert_close(f, 1.0, 1e-12, "1000 g = 1 kg");
}

#[test]
fn unit_mass_tonnes_to_kilograms() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let m = Mass::tonnes(&val);
    let f = m.eval_f64().unwrap();
    assert_close(f, 1000.0, 1e-10, "1 tonne = 1000 kg");
}

#[test]
fn unit_mass_ounces_to_kilograms() {
    let ctx = Context::new();
    let val = ctx.int(16);
    let m = Mass::ounces(&val);
    let f = m.eval_f64().unwrap();
    // 16 oz = 1 lb = 0.45359237 kg
    assert_close(f, 0.45359237, 1e-8, "16 oz = 1 lb");
}

// ---------------------------------------------------------------------------
// 1e. Unit conversions — Time
// ---------------------------------------------------------------------------

#[test]
fn unit_time_minutes_hours_days() {
    let ctx = Context::new();
    let val = ctx.int(1);

    let mins = Time::minutes(&val);
    assert_close(mins.eval_f64().unwrap(), 60.0, 1e-12, "1 min = 60 s");

    let hrs = Time::hours(&val);
    assert_close(hrs.eval_f64().unwrap(), 3600.0, 1e-10, "1 hr = 3600 s");

    let days = Time::days(&val);
    assert_close(days.eval_f64().unwrap(), 86400.0, 1e-8, "1 day = 86400 s");
}

// ---------------------------------------------------------------------------
// 1f. Unit conversions — Angle
// ---------------------------------------------------------------------------

#[test]
fn unit_angle_degrees_to_radians() {
    let ctx = Context::new();
    let val = ctx.int(180);
    let a = Angle::degrees(&val);
    let f = a.eval_f64().unwrap();
    assert_close(f, std::f64::consts::PI, 1e-10, "180° = π rad");
}

#[test]
fn unit_angle_revolutions_to_radians() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let a = Angle::revolutions(&val);
    let f = a.eval_f64().unwrap();
    assert_close(f, 2.0 * std::f64::consts::PI, 1e-10, "1 rev = 2π rad");
}

// ---------------------------------------------------------------------------
// 1g. Unit conversions — Velocity, Energy, Power, Pressure, Frequency
// ---------------------------------------------------------------------------

#[test]
fn unit_velocity_kmh_to_ms() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let v = Velocity::kilometers_per_hour(&val);
    let f = v.eval_f64().unwrap();
    assert_close(f, 1.0 / 3.6, 1e-12, "1 km/h = 1/3.6 m/s");
}

#[test]
fn unit_energy_kilowatt_hours_to_joules() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let e = Energy::kilowatt_hours(&val);
    let f = e.eval_f64().unwrap();
    assert_close(f, 3_600_000.0, 1e-4, "1 kWh = 3.6 MJ");
}

#[test]
fn unit_pressure_atmospheres_to_pascals() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let p = Pressure::atmospheres(&val);
    let f = p.eval_f64().unwrap();
    assert_close(f, 101325.0, 1e-4, "1 atm = 101325 Pa");
}

#[test]
fn unit_frequency_rpm_to_hertz() {
    let ctx = Context::new();
    let val = ctx.int(60);
    let f = Frequency::rpm(&val);
    let hz = f.eval_f64().unwrap();
    assert_close(hz, 1.0, 1e-12, "60 rpm = 1 Hz");
}

#[test]
fn unit_power_horsepower_to_watts() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let p = Power::horsepower(&val);
    let f = p.eval_f64().unwrap();
    // 1 mechanical hp ≈ 745.69987 W
    assert_close(f, 745.69987, 0.01, "1 hp ≈ 745.7 W");
}

// ---------------------------------------------------------------------------
// 1h. Unit conversions — Electrical
// ---------------------------------------------------------------------------

#[test]
fn unit_voltage_millivolts_to_volts() {
    let ctx = Context::new();
    let val = ctx.int(1000);
    let v = Voltage::millivolts(&val);
    let f = v.eval_f64().unwrap();
    assert_close(f, 1.0, 1e-12, "1000 mV = 1 V");
}

#[test]
fn unit_capacitance_microfarads() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let c = Capacitance::microfarads(&val);
    let f = c.eval_f64().unwrap();
    assert_close(f, 1e-6, 1e-18, "1 μF = 1e-6 F");
}

#[test]
fn unit_charge_ampere_hours_to_coulombs() {
    let ctx = Context::new();
    let val = ctx.int(1);
    let q = Charge::ampere_hours(&val);
    let f = q.eval_f64().unwrap();
    assert_close(f, 3600.0, 1e-8, "1 Ah = 3600 C");
}

// ---------------------------------------------------------------------------
// 1i. Named type arithmetic — compile-time dimension checking
// ---------------------------------------------------------------------------

#[test]
fn unit_arithmetic_force_equals_mass_times_acceleration() {
    let ctx = Context::new();
    let m = Mass::symbol(&ctx, "m");
    let a = Acceleration::symbol(&ctx, "a");
    // dim! macro does compile-time type checking
    let f: Force = symplex::dim!(ctx, Force: m * a);
    let f_num = f
        .subs(&m, &ctx.int(10))
        .subs(&a, &ctx.rational(981, 100))
        .eval();
    assert_close(f_num.eval_f64().unwrap(), 98.1, 1e-10, "F = 10 * 9.81");
}

#[test]
fn unit_arithmetic_energy_equals_force_times_distance() {
    let ctx = Context::new();
    let f = Force::symbol(&ctx, "F");
    let d = Length::symbol(&ctx, "d");
    let w: Energy = symplex::dim!(ctx, Energy: f * d);
    let w_num = w.subs(&f, &ctx.int(100)).subs(&d, &ctx.int(5)).eval();
    assert_close(w_num.eval_f64().unwrap(), 500.0, 1e-10, "W = 100 * 5");
}

#[test]
fn unit_arithmetic_power_equals_energy_over_time() {
    let ctx = Context::new();
    let e = Energy::symbol(&ctx, "E");
    let t = Time::symbol(&ctx, "t");
    let p: Power = symplex::dim!(ctx, Power: e / t);
    let p_num = p.subs(&e, &ctx.int(1000)).subs(&t, &ctx.int(10)).eval();
    assert_close(p_num.eval_f64().unwrap(), 100.0, 1e-10, "P = 1000/10");
}

#[test]
fn unit_arithmetic_ohms_law_voltage() {
    let ctx = Context::new();
    let i = Current::symbol(&ctx, "I");
    let r = Resistance::symbol(&ctx, "R");
    let v: Voltage = symplex::dim!(ctx, Voltage: i * r);
    let v_num = v.subs(&i, &ctx.int(3)).subs(&r, &ctx.int(100)).eval();
    assert_close(v_num.eval_f64().unwrap(), 300.0, 1e-10, "V = 3 * 100");
}

#[test]
fn unit_add_same_dimension_works() {
    let ctx = Context::new();
    let a = Length::constant(&ctx, 3);
    let b = Length::constant(&ctx, 7);
    let c = &a + &b;
    assert_close(c.eval_f64().unwrap(), 10.0, 1e-12, "3m + 7m = 10m");
}

#[test]
fn unit_sub_same_dimension_works() {
    let ctx = Context::new();
    let a = Force::constant(&ctx, 50);
    let b = Force::constant(&ctx, 20);
    let c = &a - &b;
    assert_close(c.eval_f64().unwrap(), 30.0, 1e-12, "50N - 20N = 30N");
}

#[test]
fn unit_neg_works() {
    let ctx = Context::new();
    let a = Velocity::constant(&ctx, 5);
    let neg_a = -&a;
    assert_close(neg_a.eval_f64().unwrap(), -5.0, 1e-12, "-5 m/s");
}

#[test]
fn unit_scalar_mul_i64() {
    let ctx = Context::new();
    let m = Mass::constant(&ctx, 3);
    let doubled = &m * 2;
    assert_close(doubled.eval_f64().unwrap(), 6.0, 1e-12, "3 kg * 2 = 6 kg");
}

#[test]
fn unit_scalar_div_i64() {
    let ctx = Context::new();
    let e = Energy::constant(&ctx, 100);
    let half = e / 2;
    assert_close(half.eval_f64().unwrap(), 50.0, 1e-12, "100 J / 2 = 50 J");
}

// ---------------------------------------------------------------------------
// 1j. Angle trig functions
// ---------------------------------------------------------------------------

#[test]
fn unit_angle_sin_cos_tan() {
    let ctx = Context::new();
    let quarter = ctx.rational(1, 4);
    let pi = ctx.pi();
    let angle_ex = &quarter * &pi; // π/4
    let angle = Angle::from_ex(angle_ex);

    let s: Dimensionless = angle.sin();
    let c: Dimensionless = angle.cos();
    let t: Dimensionless = angle.tan();

    let sv = s.eval_f64().unwrap();
    let cv = c.eval_f64().unwrap();
    let tv = t.eval_f64().unwrap();

    let sqrt2_over_2 = std::f64::consts::FRAC_1_SQRT_2;
    assert_close(sv, sqrt2_over_2, 1e-10, "sin(π/4)");
    assert_close(cv, sqrt2_over_2, 1e-10, "cos(π/4)");
    assert_close(tv, 1.0, 1e-10, "tan(π/4)");
}

// ---------------------------------------------------------------------------
// 1k. Cross-type conversions
// ---------------------------------------------------------------------------

#[test]
fn unit_energy_to_torque_from() {
    let ctx = Context::new();
    let e = Energy::constant(&ctx, 42);
    let t: Torque = Torque::from(e);
    assert_close(t.eval_f64().unwrap(), 42.0, 1e-12, "Energy → Torque");
}

#[test]
fn unit_torque_to_energy_from() {
    let ctx = Context::new();
    let t = Torque::from_energy(Energy::constant(&ctx, 99));
    assert_close(t.eval_f64().unwrap(), 99.0, 1e-12, "Torque from Energy");
}

#[test]
fn unit_frequency_angular_velocity_roundtrip() {
    let ctx = Context::new();
    let f = Frequency::from_angular_velocity(AngularVelocity::constant(&ctx, 10));
    assert_close(f.eval_f64().unwrap(), 10.0, 1e-12, "Freq from AngVel");
}

// ---------------------------------------------------------------------------
// 1l. Temperature conversions
// ---------------------------------------------------------------------------

#[test]
fn unit_temperature_celsius_to_kelvin() {
    let ctx = Context::new();
    let val = ctx.int(100);
    let t = Temperature::from_celsius(&val);
    let k = t.eval_f64().unwrap();
    assert_close(k, 373.15, 1e-8, "100°C = 373.15 K");
}

#[test]
fn unit_temperature_fahrenheit_boiling() {
    let ctx = Context::new();
    let val = ctx.int(212);
    let t = Temperature::from_fahrenheit(&val);
    let k = t.eval_f64().unwrap();
    assert_close(k, 373.15, 0.1, "212°F ≈ 373.15 K");
}

#[test]
fn unit_temperature_absolute_zero_rankine() {
    let ctx = Context::new();
    let val = ctx.int(0);
    let t = Temperature::from_rankine(&val);
    let k = t.eval_f64().unwrap();
    assert_close(k, 0.0, 1e-10, "0°R = 0 K");
}

// ---------------------------------------------------------------------------
// 1m. Runtime dimension inference
// ---------------------------------------------------------------------------

#[test]
fn unit_runtime_dimension_inference_correct() {
    let ctx = Context::new();
    let m_sym = ctx.symbol("m");
    let a_sym = ctx.symbol("a");

    let dims = DimMap::new()
        .with("m", ConstDim::MASS)
        .with("a", ConstDim::ACCELERATION);

    // m * a should infer as Force
    let force_ex = &m_sym * &a_sym;
    let inferred = infer_dimension(&force_ex, &dims).unwrap();
    assert!(
        inferred.eq(ConstDim::FORCE),
        "m*a should infer as Force, got {:?}",
        inferred
    );
}

#[test]
fn unit_runtime_dimension_inference_mismatch() {
    let ctx = Context::new();
    let m_sym = ctx.symbol("m");
    let a_sym = ctx.symbol("a");

    let dims = DimMap::new()
        .with("m", ConstDim::MASS)
        .with("a", ConstDim::ACCELERATION);

    let force_ex = &m_sym * &a_sym;
    // Trying to create a Velocity from a Force expression should fail
    let result = Velocity::checked_from_ex(force_ex, &dims);
    assert!(
        result.is_err(),
        "Force expression misidentified as Velocity"
    );
}

// ---------------------------------------------------------------------------
// 1n. Volume conversions
// ---------------------------------------------------------------------------

#[test]
fn unit_volume_liters_to_cubic_meters() {
    let ctx = Context::new();
    let val = ctx.int(1000);
    let v = Volume::liters(&val);
    let f = v.eval_f64().unwrap();
    assert_close(f, 1.0, 1e-10, "1000 L = 1 m³");
}

// ---------------------------------------------------------------------------
// 1o. Dimensionless percent and per_mille
// ---------------------------------------------------------------------------

#[test]
fn unit_dimensionless_percent() {
    let ctx = Context::new();
    let val = ctx.int(50);
    let d = Dimensionless::percent(&val);
    let f = d.eval_f64().unwrap();
    assert_close(f, 0.5, 1e-12, "50% = 0.5");
}

#[test]
fn unit_dimensionless_per_mille() {
    let ctx = Context::new();
    let val = ctx.int(100);
    let d = Dimensionless::per_mille(&val);
    let f = d.eval_f64().unwrap();
    assert_close(f, 0.1, 1e-12, "100‰ = 0.1");
}

// ---------------------------------------------------------------------------
// 1p. Symbolic units — variables survive conversions
// ---------------------------------------------------------------------------

#[test]
fn unit_symbolic_length_preserves_variable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let l = Length::kilometers(&x);
    // Should contain x in its expression
    assert!(
        l.contains(&x),
        "symbolic length should contain the variable x"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. MATRIX OPERATIONS — EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

// ---------------------------------------------------------------------------
// 2a. 1×1 matrices
// ---------------------------------------------------------------------------

#[test]
fn matrix_1x1_det() {
    let ctx = Context::new();
    let m = Matrix::new(vec![vec![ctx.int(7)]]).unwrap();
    let d = m.det().unwrap();
    assert_close(d.eval_f64().unwrap(), 7.0, 1e-12, "det([[7]]) = 7");
}

#[test]
fn matrix_1x1_inv() {
    let ctx = Context::new();
    let m = Matrix::new(vec![vec![ctx.int(4)]]).unwrap();
    let inv = m.inv().unwrap();
    assert_close(
        inv.get(0, 0).eval_f64().unwrap(),
        0.25,
        1e-12,
        "inv([[4]]) = [[0.25]]",
    );
}

#[test]
fn matrix_1x1_trace() {
    let ctx = Context::new();
    let m = Matrix::new(vec![vec![ctx.int(42)]]).unwrap();
    let t = m.trace().unwrap();
    assert_close(t.eval_f64().unwrap(), 42.0, 1e-12, "trace([[42]]) = 42");
}

#[test]
fn matrix_1x1_matmul() {
    let ctx = Context::new();
    let a = Matrix::new(vec![vec![ctx.int(3)]]).unwrap();
    let b = Matrix::new(vec![vec![ctx.int(5)]]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_close(
        c.get(0, 0).eval_f64().unwrap(),
        15.0,
        1e-12,
        "[[3]] * [[5]] = [[15]]",
    );
}

#[test]
fn matrix_1x1_transpose() {
    let ctx = Context::new();
    let m = Matrix::new(vec![vec![ctx.int(9)]]).unwrap();
    let t = m.transpose();
    assert_eq!(t.shape(), (1, 1));
    assert_close(t.get(0, 0).eval_f64().unwrap(), 9.0, 1e-12, "transpose 1×1");
}

// ---------------------------------------------------------------------------
// 2b. Larger matrices: 4×4, 5×5, 6×6
// ---------------------------------------------------------------------------

#[test]
fn matrix_4x4_det_numeric() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3), ctx.int(4)],
        vec![ctx.int(5), ctx.int(6), ctx.int(7), ctx.int(8)],
        vec![ctx.int(9), ctx.int(10), ctx.int(11), ctx.int(12)],
        vec![ctx.int(13), ctx.int(14), ctx.int(15), ctx.int(16)],
    ])
    .unwrap();
    // This matrix is singular (rows are arithmetic progressions)
    let d = m.det().unwrap().eval();
    assert_close(d.eval_f64().unwrap(), 0.0, 1e-10, "det of singular 4×4 = 0");
}

#[test]
fn matrix_4x4_det_nonsingular() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(2), ctx.int(1), ctx.int(0), ctx.int(0)],
        vec![ctx.int(1), ctx.int(3), ctx.int(1), ctx.int(0)],
        vec![ctx.int(0), ctx.int(1), ctx.int(4), ctx.int(1)],
        vec![ctx.int(0), ctx.int(0), ctx.int(1), ctx.int(5)],
    ])
    .unwrap();
    // Tridiagonal matrix — det via recurrence:
    // d0=1, d1=2, d2=3·2−1=5, d3=4·5−2=18, d4=5·18−5=85
    let d = m.det().unwrap().eval();
    assert_close(d.eval_f64().unwrap(), 85.0, 1e-8, "det of tridiagonal 4×4");
}

#[test]
fn matrix_5x5_identity_det() {
    let ctx = Context::new();
    let m = Matrix::identity(&ctx, 5).unwrap();
    let d = m.det().unwrap().eval();
    assert_close(d.eval_f64().unwrap(), 1.0, 1e-12, "det(I_5) = 1");
}

#[test]
fn matrix_5x5_diagonal_det() {
    let ctx = Context::new();
    let diag = vec![ctx.int(2), ctx.int(3), ctx.int(4), ctx.int(5), ctx.int(6)];
    let m = Matrix::diag(&diag).unwrap();
    let d = m.det().unwrap().eval();
    // det = 2 * 3 * 4 * 5 * 6 = 720
    assert_close(d.eval_f64().unwrap(), 720.0, 1e-8, "det of 5×5 diagonal");
}

#[test]
fn matrix_6x6_identity_inv() {
    let ctx = Context::new();
    let m = Matrix::identity(&ctx, 6).unwrap();
    let inv = m.inv().unwrap();
    // I⁻¹ = I
    for i in 0..6 {
        for j in 0..6 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(
                inv.get(i, j).eval_f64().unwrap(),
                expected,
                1e-12,
                &format!("I⁻¹[{i},{j}]"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2c. Singular matrix — inverse should error
// ---------------------------------------------------------------------------

#[test]
fn matrix_singular_inv_errors() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(2), ctx.int(4)], // Row 2 = 2 * Row 1
    ])
    .unwrap();
    let result = m.inv();
    assert!(result.is_err(), "Inverse of singular matrix should error");
}

#[test]
fn matrix_singular_3x3_inv_errors() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
        vec![ctx.int(7), ctx.int(8), ctx.int(9)],
    ])
    .unwrap();
    // This matrix has det = 0
    let result = m.inv();
    assert!(result.is_err(), "Inverse of singular 3×3 should error");
}

// ---------------------------------------------------------------------------
// 2d. Non-square matrices
// ---------------------------------------------------------------------------

#[test]
fn matrix_non_square_transpose() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    assert_eq!(m.shape(), (2, 3));
    let t = m.transpose();
    assert_eq!(t.shape(), (3, 2));
    assert_close(
        t.get(2, 0).eval_f64().unwrap(),
        3.0,
        1e-12,
        "transpose (2,0)",
    );
    assert_close(
        t.get(0, 1).eval_f64().unwrap(),
        4.0,
        1e-12,
        "transpose (0,1)",
    );
}

#[test]
fn matrix_non_square_matmul() {
    let ctx = Context::new();
    // 2×3 * 3×2 = 2×2
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    let b = Matrix::new(vec![
        vec![ctx.int(7), ctx.int(8)],
        vec![ctx.int(9), ctx.int(10)],
        vec![ctx.int(11), ctx.int(12)],
    ])
    .unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), (2, 2));
    // c[0][0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
    assert_close(c.get(0, 0).eval_f64().unwrap(), 58.0, 1e-10, "matmul [0,0]");
    // c[0][1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64
    assert_close(c.get(0, 1).eval_f64().unwrap(), 64.0, 1e-10, "matmul [0,1]");
    // c[1][0] = 4*7 + 5*9 + 6*11 = 28+45+66 = 139
    assert_close(
        c.get(1, 0).eval_f64().unwrap(),
        139.0,
        1e-10,
        "matmul [1,0]",
    );
    // c[1][1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
    assert_close(
        c.get(1, 1).eval_f64().unwrap(),
        154.0,
        1e-10,
        "matmul [1,1]",
    );
}

#[test]
fn matrix_non_square_det_errors() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    let result = m.det();
    assert!(result.is_err(), "det of non-square matrix should error");
}

#[test]
fn matrix_non_square_inv_errors() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    let result = m.inv();
    assert!(result.is_err(), "inv of non-square matrix should error");
}

// ---------------------------------------------------------------------------
// 2e. LU decomposition
// ---------------------------------------------------------------------------

#[test]
fn matrix_lu_2x2() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(4), ctx.int(3)],
        vec![ctx.int(6), ctx.int(3)],
    ])
    .unwrap();
    let Lu { l, u, perm } = m.lu().expect("LU should succeed for non-singular 2×2");
    // Verify P*A = L*U by reconstructing
    let lu = l.matmul(&u).unwrap();
    for (i, &orig_row) in perm.iter().enumerate().take(2) {
        for j in 0..2 {
            let expected = m.get(orig_row, j).eval_f64().unwrap();
            let actual = lu.get(i, j).eval().eval_f64().unwrap();
            assert_close(actual, expected, 1e-10, &format!("LU[{i},{j}]"));
        }
    }
}

#[test]
fn matrix_lu_3x3() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(2), ctx.int(1), ctx.int(1)],
        vec![ctx.int(4), ctx.int(3), ctx.int(3)],
        vec![ctx.int(8), ctx.int(7), ctx.int(9)],
    ])
    .unwrap();
    let Lu { l, u, perm } = m.lu().expect("LU should succeed");
    let lu = l.matmul(&u).unwrap();
    for (i, &orig_row) in perm.iter().enumerate().take(3) {
        for j in 0..3 {
            let expected = m.get(orig_row, j).eval_f64().unwrap();
            let actual = lu.get(i, j).eval().eval_f64().unwrap();
            assert_close(actual, expected, 1e-10, &format!("LU 3×3 [{i},{j}]"));
        }
    }
}

#[test]
fn matrix_lu_singular_returns_none() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(2), ctx.int(4)],
    ])
    .unwrap();
    assert!(m.lu().is_err(), "LU of singular matrix should return Err");
}

// ---------------------------------------------------------------------------
// 2f. Cholesky decomposition
// ---------------------------------------------------------------------------

#[test]
fn matrix_cholesky_positive_definite() {
    let ctx = Context::new();
    // A = [[4,2],[2,3]] — symmetric positive definite
    let m = Matrix::new(vec![
        vec![ctx.int(4), ctx.int(2)],
        vec![ctx.int(2), ctx.int(3)],
    ])
    .unwrap();
    let l = m
        .cholesky()
        .expect("Cholesky should succeed for SPD matrix");

    // Verify L * Lᵀ = A
    let lt = l.transpose();
    let product = l.matmul(&lt).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let expected = m.get(i, j).eval_f64().unwrap();
            let actual = product.get(i, j).eval().eval_f64().unwrap();
            assert_close(actual, expected, 1e-10, &format!("LLᵀ[{i},{j}]"));
        }
    }
}

#[test]
fn matrix_cholesky_3x3_spd() {
    let ctx = Context::new();
    // A = [[4,2,0],[2,5,2],[0,2,6]] — symmetric positive definite
    let m = Matrix::new(vec![
        vec![ctx.int(4), ctx.int(2), ctx.int(0)],
        vec![ctx.int(2), ctx.int(5), ctx.int(2)],
        vec![ctx.int(0), ctx.int(2), ctx.int(6)],
    ])
    .unwrap();
    let l = m.cholesky().expect("Cholesky should succeed for 3×3 SPD");
    let lt = l.transpose();
    let product = l.matmul(&lt).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let expected = m.get(i, j).eval_f64().unwrap();
            let actual = product.get(i, j).eval().eval_f64().unwrap();
            assert_close(actual, expected, 1e-8, &format!("LLᵀ 3×3 [{i},{j}]"));
        }
    }
}

#[test]
fn matrix_cholesky_not_positive_definite_returns_none() {
    let ctx = Context::new();
    // A = [[1, 2], [2, 1]] — eigenvalues are 3 and -1, not PD
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(2), ctx.int(1)],
    ])
    .unwrap();
    let result = m.cholesky();
    assert!(
        result.is_err(),
        "Cholesky should return Err for non-PD matrix"
    );
}

#[test]
fn matrix_cholesky_non_square_errors() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    let result = m.cholesky();
    assert!(
        result.is_err(),
        "Cholesky of non-square matrix should error"
    );
}

// ---------------------------------------------------------------------------
// 2g. Pseudo-inverse
// ---------------------------------------------------------------------------

#[test]
fn matrix_pinv_square_nonsingular() {
    let ctx = Context::new();
    // For a square nonsingular matrix, pinv = inv
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3), ctx.int(4)],
    ])
    .unwrap();
    let inv = m.inv().unwrap();
    let pinv = m.pinv().unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let inv_val = inv.get(i, j).eval().eval_f64().unwrap();
            let pinv_val = pinv.get(i, j).eval().eval_f64().unwrap();
            assert_close(pinv_val, inv_val, 1e-8, &format!("pinv vs inv [{i},{j}]"));
        }
    }
}

// ---------------------------------------------------------------------------
// 2h. Matrix exponential
// ---------------------------------------------------------------------------

#[test]
fn matrix_exp_zero_matrix() {
    let ctx = Context::new();
    let z = Matrix::zeros(&ctx, 2, 2).unwrap();
    let result = z.matrix_exp().unwrap();
    // e^0 = I
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let actual = result.get(i, j).eval().eval_f64().unwrap();
            assert_close(actual, expected, 1e-10, &format!("exp(0)[{i},{j}]"));
        }
    }
}

#[test]
fn matrix_exp_identity_matrix() {
    let ctx = Context::new();
    let id = Matrix::identity(&ctx, 2).unwrap();
    let result = id.matrix_exp().unwrap();
    // e^I = e * I
    let e_val = std::f64::consts::E;
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { e_val } else { 0.0 };
            let actual = result.get(i, j).eval().eval_f64().unwrap();
            assert_close(actual, expected, 1e-6, &format!("exp(I)[{i},{j}]"));
        }
    }
}

#[test]
fn matrix_exp_non_square_errors() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    let result = m.matrix_exp();
    assert!(result.is_err(), "matrix_exp of non-square should error");
}

#[test]
fn matrix_exp_series_diagonal() {
    let ctx = Context::new();
    // exp of diagonal matrix [[a,0],[0,b]] = [[e^a,0],[0,e^b]]
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(0)],
        vec![ctx.int(0), ctx.int(2)],
    ])
    .unwrap();
    let result = m.exp_series(15).unwrap();
    let e1 = std::f64::consts::E;
    let e2 = e1 * e1;
    assert_close(
        result.get(0, 0).eval().eval_f64().unwrap(),
        e1,
        1e-6,
        "exp_series diag [0,0]",
    );
    assert_close(
        result.get(1, 1).eval().eval_f64().unwrap(),
        e2,
        1e-4,
        "exp_series diag [1,1]",
    );
    assert_close(
        result.get(0, 1).eval().eval_f64().unwrap(),
        0.0,
        1e-8,
        "exp_series diag [0,1]",
    );
}

// ---------------------------------------------------------------------------
// 2i. Rank, nullspace, columnspace
// ---------------------------------------------------------------------------

#[test]
fn matrix_rank_full_rank() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3), ctx.int(4)],
    ])
    .unwrap();
    assert_eq!(m.rank(), 2);
}

#[test]
fn matrix_rank_deficient() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(2), ctx.int(4)],
    ])
    .unwrap();
    assert_eq!(m.rank(), 1);
}

#[test]
fn matrix_nullspace_of_identity_is_empty() {
    let ctx = Context::new();
    let m = Matrix::identity(&ctx, 3).unwrap();
    let ns = m.nullspace();
    assert!(ns.is_empty(), "nullspace of identity should be empty");
}

#[test]
fn matrix_nullspace_of_rank_deficient() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(2), ctx.int(4)],
    ])
    .unwrap();
    let ns = m.nullspace();
    assert_eq!(ns.len(), 1, "rank-1 2×2 matrix should have 1D nullspace");
}

#[test]
fn matrix_columnspace_dimension() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
        vec![ctx.int(7), ctx.int(8), ctx.int(9)],
    ])
    .unwrap();
    let cs = m.columnspace();
    // Rank should be 2
    assert_eq!(cs.len(), 2, "rank-2 3×3 should have 2D column space");
}

// ---------------------------------------------------------------------------
// 2j. RREF
// ---------------------------------------------------------------------------

#[test]
fn matrix_rref_identity() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(0), ctx.int(0)],
        vec![ctx.int(0), ctx.int(1), ctx.int(0)],
        vec![ctx.int(0), ctx.int(0), ctx.int(1)],
    ])
    .unwrap();
    let (rref, pivots) = m.rref();
    assert_eq!(pivots, vec![0, 1, 2]);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(
                rref.get(i, j).eval_f64().unwrap(),
                expected,
                1e-12,
                &format!("rref I[{i},{j}]"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2k. Matrix solve
// ---------------------------------------------------------------------------

#[test]
fn matrix_solve_2x2() {
    let ctx = Context::new();
    // 2x + y = 5
    // x + 3y = 10
    let a = Matrix::new(vec![
        vec![ctx.int(2), ctx.int(1)],
        vec![ctx.int(1), ctx.int(3)],
    ])
    .unwrap();
    let b = Matrix::new(vec![vec![ctx.int(5)], vec![ctx.int(10)]]).unwrap();
    let x = a.solve(&b).unwrap();
    assert_eq!(x.shape(), (2, 1));
    // x = 1, y = 3
    assert_close(x.get(0, 0).eval_f64().unwrap(), 1.0, 1e-10, "x = 1");
    assert_close(x.get(1, 0).eval_f64().unwrap(), 3.0, 1e-10, "y = 3");
}

#[test]
fn matrix_solve_singular_errors() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(2), ctx.int(4)],
    ])
    .unwrap();
    let b = Matrix::new(vec![vec![ctx.int(3)], vec![ctx.int(6)]]).unwrap();
    let result = a.solve(&b);
    assert!(result.is_err(), "solve with singular matrix should error");
}

// ---------------------------------------------------------------------------
// 2l. Kronecker product
// ---------------------------------------------------------------------------

#[test]
fn matrix_kronecker_2x2() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3), ctx.int(4)],
    ])
    .unwrap();
    let b = Matrix::identity(&ctx, 2).unwrap();
    let k = a.kronecker(&b);
    assert_eq!(k.shape(), (4, 4));
    // A ⊗ I₂ has 2×2 blocks: [[1·I, 2·I], [3·I, 4·I]]
    assert_close(k.get(0, 0).eval_f64().unwrap(), 1.0, 1e-12, "kron [0,0]");
    assert_close(k.get(0, 1).eval_f64().unwrap(), 0.0, 1e-12, "kron [0,1]");
    assert_close(k.get(0, 2).eval_f64().unwrap(), 2.0, 1e-12, "kron [0,2]");
    // K = [[1·I, 2·I], [3·I, 4·I]] = [[1,0,2,0],[0,1,0,2],[3,0,4,0],[0,3,0,4]]
    assert_close(k.get(1, 3).eval_f64().unwrap(), 2.0, 1e-12, "kron [1,3]");
}

// ---------------------------------------------------------------------------
// 2m. Matrix power (powi)
// ---------------------------------------------------------------------------

#[test]
fn matrix_powi_zero_gives_identity() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(2), ctx.int(1)],
        vec![ctx.int(0), ctx.int(3)],
    ])
    .unwrap();
    let m0 = m.powi(0).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(
                m0.get(i, j).eval_f64().unwrap(),
                expected,
                1e-12,
                &format!("A^0[{i},{j}]"),
            );
        }
    }
}

#[test]
fn matrix_powi_one_gives_self() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(2), ctx.int(1)],
        vec![ctx.int(0), ctx.int(3)],
    ])
    .unwrap();
    let m1 = m.powi(1).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert_close(
                m1.get(i, j).eval().eval_f64().unwrap(),
                m.get(i, j).eval_f64().unwrap(),
                1e-12,
                &format!("A^1[{i},{j}]"),
            );
        }
    }
}

#[test]
fn matrix_powi_two_equals_matmul_self() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3), ctx.int(4)],
    ])
    .unwrap();
    let m2 = m.powi(2).unwrap();
    let m_sq = m.matmul(&m).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert_close(
                m2.get(i, j).eval().eval_f64().unwrap(),
                m_sq.get(i, j).eval().eval_f64().unwrap(),
                1e-10,
                &format!("A^2[{i},{j}]"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2n. Symmetry check
// ---------------------------------------------------------------------------

#[test]
fn matrix_is_symmetric_true() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(2), ctx.int(5), ctx.int(6)],
        vec![ctx.int(3), ctx.int(6), ctx.int(9)],
    ])
    .unwrap();
    assert_eq!(m.is_symmetric(), Some(true));
}

#[test]
fn matrix_is_symmetric_false() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3), ctx.int(4)],
    ])
    .unwrap();
    assert_eq!(m.is_symmetric(), Some(false));
}

// ---------------------------------------------------------------------------
// 2o. Norm
// ---------------------------------------------------------------------------

#[test]
fn matrix_frobenius_norm() {
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3), ctx.int(4)],
    ])
    .unwrap();
    let n = m.norm().eval();
    // ||M||_F = sqrt(1+4+9+16) = sqrt(30)
    assert_close(
        n.eval_f64().unwrap(),
        30.0_f64.sqrt(),
        1e-10,
        "Frobenius norm",
    );
}

// ---------------------------------------------------------------------------
// 2p. hstack / vstack
// ---------------------------------------------------------------------------

#[test]
fn matrix_hstack_two_matrices() {
    let ctx = Context::new();
    let a = Matrix::new(vec![vec![ctx.int(1)], vec![ctx.int(2)]]).unwrap();
    let b = Matrix::new(vec![vec![ctx.int(3)], vec![ctx.int(4)]]).unwrap();
    let h = Matrix::hstack(&[&a, &b]).unwrap();
    assert_eq!(h.shape(), (2, 2));
    assert_close(h.get(0, 1).eval_f64().unwrap(), 3.0, 1e-12, "hstack [0,1]");
}

#[test]
fn matrix_vstack_two_matrices() {
    let ctx = Context::new();
    let a = Matrix::new(vec![vec![ctx.int(1), ctx.int(2)]]).unwrap();
    let b = Matrix::new(vec![vec![ctx.int(3), ctx.int(4)]]).unwrap();
    let v = Matrix::vstack(&[&a, &b]).unwrap();
    assert_eq!(v.shape(), (2, 2));
    assert_close(v.get(1, 0).eval_f64().unwrap(), 3.0, 1e-12, "vstack [1,0]");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. QUATERNION OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

use symplex::quaternion::Quaternion;

// ---------------------------------------------------------------------------
// 3a. Non-commutativity: i*j ≠ j*i
// ---------------------------------------------------------------------------

#[test]
fn quaternion_multiplication_non_commutative() {
    let ctx = Context::new();
    // q1 = i (pure imaginary)
    let qi = Quaternion::new(ctx.int(0), ctx.int(1), ctx.int(0), ctx.int(0));
    // q2 = j
    let qj = Quaternion::new(ctx.int(0), ctx.int(0), ctx.int(1), ctx.int(0));

    // i * j = k
    let ij = qi.mul(&qj);
    let (w1, x1, y1, z1) = quat_to_f64(&ij.eval());
    assert_close(w1, 0.0, 1e-12, "ij w");
    assert_close(x1, 0.0, 1e-12, "ij x");
    assert_close(y1, 0.0, 1e-12, "ij y");
    assert_close(z1, 1.0, 1e-12, "ij z");

    // j * i = -k
    let ji = qj.mul(&qi);
    let (w2, x2, y2, z2) = quat_to_f64(&ji.eval());
    assert_close(w2, 0.0, 1e-12, "ji w");
    assert_close(x2, 0.0, 1e-12, "ji x");
    assert_close(y2, 0.0, 1e-12, "ji y");
    assert_close(z2, -1.0, 1e-12, "ji z");

    // Therefore i*j ≠ j*i
}

#[test]
fn quaternion_ijk_equals_minus_one() {
    let ctx = Context::new();
    let qi = Quaternion::new(ctx.int(0), ctx.int(1), ctx.int(0), ctx.int(0));
    let qj = Quaternion::new(ctx.int(0), ctx.int(0), ctx.int(1), ctx.int(0));
    let qk = Quaternion::new(ctx.int(0), ctx.int(0), ctx.int(0), ctx.int(1));

    // Hamilton's identity: i*j*k = -1
    let ijk = qi.mul(&qj).mul(&qk);
    let (w, x, y, z) = quat_to_f64(&ijk.eval());
    assert_close(w, -1.0, 1e-12, "ijk = -1 (w)");
    assert_close(x, 0.0, 1e-12, "ijk = -1 (x)");
    assert_close(y, 0.0, 1e-12, "ijk = -1 (y)");
    assert_close(z, 0.0, 1e-12, "ijk = -1 (z)");
}

#[test]
fn quaternion_i_squared_equals_minus_one() {
    let ctx = Context::new();
    let qi = Quaternion::new(ctx.int(0), ctx.int(1), ctx.int(0), ctx.int(0));
    let ii = qi.mul(&qi);
    let (w, x, y, z) = quat_to_f64(&ii.eval());
    assert_close(w, -1.0, 1e-12, "i² = -1 (w)");
    assert_close(x, 0.0, 1e-12, "i² (x)");
    assert_close(y, 0.0, 1e-12, "i² (y)");
    assert_close(z, 0.0, 1e-12, "i² (z)");
}

#[test]
fn quaternion_j_squared_equals_minus_one() {
    let ctx = Context::new();
    let qj = Quaternion::new(ctx.int(0), ctx.int(0), ctx.int(1), ctx.int(0));
    let jj = qj.mul(&qj);
    let (w, _, _, _) = quat_to_f64(&jj.eval());
    assert_close(w, -1.0, 1e-12, "j² = -1");
}

#[test]
fn quaternion_k_squared_equals_minus_one() {
    let ctx = Context::new();
    let qk = Quaternion::new(ctx.int(0), ctx.int(0), ctx.int(0), ctx.int(1));
    let kk = qk.mul(&qk);
    let (w, _, _, _) = quat_to_f64(&kk.eval());
    assert_close(w, -1.0, 1e-12, "k² = -1");
}

// ---------------------------------------------------------------------------
// 3b. Conjugate
// ---------------------------------------------------------------------------

#[test]
fn quaternion_conjugate() {
    let ctx = Context::new();
    let q = Quaternion::new(ctx.int(1), ctx.int(2), ctx.int(3), ctx.int(4));
    let conj = q.conjugate();
    let (w, x, y, z) = quat_to_f64(&conj);
    assert_close(w, 1.0, 1e-12, "conj w");
    assert_close(x, -2.0, 1e-12, "conj x");
    assert_close(y, -3.0, 1e-12, "conj y");
    assert_close(z, -4.0, 1e-12, "conj z");
}

#[test]
fn quaternion_double_conjugate_is_identity() {
    let ctx = Context::new();
    let q = Quaternion::new(ctx.int(1), ctx.int(2), ctx.int(3), ctx.int(4));
    let dconj = q.conjugate().conjugate();
    let (w, x, y, z) = quat_to_f64(&dconj.eval());
    assert_close(w, 1.0, 1e-12, "q** w");
    assert_close(x, 2.0, 1e-12, "q** x");
    assert_close(y, 3.0, 1e-12, "q** y");
    assert_close(z, 4.0, 1e-12, "q** z");
}

// ---------------------------------------------------------------------------
// 3c. Norm
// ---------------------------------------------------------------------------

#[test]
fn quaternion_norm() {
    let ctx = Context::new();
    let q = Quaternion::new(ctx.int(1), ctx.int(2), ctx.int(3), ctx.int(4));
    let n = q.norm().eval();
    // |q| = sqrt(1+4+9+16) = sqrt(30)
    assert_close(n.eval_f64().unwrap(), 30.0_f64.sqrt(), 1e-10, "|q|");
}

#[test]
fn quaternion_norm_squared() {
    let ctx = Context::new();
    let q = Quaternion::new(ctx.int(1), ctx.int(2), ctx.int(3), ctx.int(4));
    let n2 = q.norm_squared().eval();
    assert_close(n2.eval_f64().unwrap(), 30.0, 1e-10, "|q|²");
}

#[test]
fn quaternion_unit_quaternion_has_norm_one() {
    let ctx = Context::new();
    let q = Quaternion::new(ctx.int(1), ctx.int(2), ctx.int(3), ctx.int(4));
    let u = q.normalize();
    let n = u.norm().eval().simplify();
    assert_close(n.eval_f64().unwrap(), 1.0, 1e-10, "normalized |q| = 1");
}

// ---------------------------------------------------------------------------
// 3d. Inverse
// ---------------------------------------------------------------------------

#[test]
fn quaternion_inverse_mul_gives_identity() {
    let ctx = Context::new();
    let q = Quaternion::new(ctx.int(1), ctx.int(2), ctx.int(3), ctx.int(4));
    let inv = q.inverse();
    let product = q.mul(&inv);

    let (w, x, y, z) = quat_to_f64(&product.eval().simplify());
    assert_close(w, 1.0, 1e-8, "q*q⁻¹ w");
    assert_close(x, 0.0, 1e-8, "q*q⁻¹ x");
    assert_close(y, 0.0, 1e-8, "q*q⁻¹ y");
    assert_close(z, 0.0, 1e-8, "q*q⁻¹ z");
}

#[test]
fn quaternion_left_inverse_mul_gives_identity() {
    let ctx = Context::new();
    let q = Quaternion::new(ctx.int(1), ctx.int(2), ctx.int(3), ctx.int(4));
    let inv = q.inverse();
    let product = inv.mul(&q);

    let (w, x, y, z) = quat_to_f64(&product.eval().simplify());
    assert_close(w, 1.0, 1e-8, "q⁻¹*q w");
    assert_close(x, 0.0, 1e-8, "q⁻¹*q x");
    assert_close(y, 0.0, 1e-8, "q⁻¹*q y");
    assert_close(z, 0.0, 1e-8, "q⁻¹*q z");
}

// ---------------------------------------------------------------------------
// 3e. Rotation — axis-angle
// ---------------------------------------------------------------------------

#[test]
fn quaternion_from_axis_angle_90deg_z() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let pi = ctx.pi();
    let half_pi = &pi / 2;

    // 90° rotation about z-axis
    let q = Quaternion::from_axis_angle(&zero, &zero, &one, &half_pi);
    let (w, x, y, z) = quat_to_f64(&q.eval());
    // q = cos(π/4) + sin(π/4)·k
    let c = std::f64::consts::FRAC_1_SQRT_2;
    assert_close(w, c, 1e-10, "axis-angle w");
    assert_close(x, 0.0, 1e-10, "axis-angle x");
    assert_close(y, 0.0, 1e-10, "axis-angle y");
    assert_close(z, c, 1e-10, "axis-angle z");
}

// ---------------------------------------------------------------------------
// 3f. Rotation matrix conversion
// ---------------------------------------------------------------------------

#[test]
fn quaternion_identity_to_rotation_matrix_is_identity() {
    let ctx = Context::new();
    let id = Quaternion::identity(&ctx);
    let r = id.to_rotation_matrix();
    assert_eq!(r.shape(), (3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(
                r.get(i, j).eval().eval_f64().unwrap(),
                expected,
                1e-10,
                &format!("R_id[{i},{j}]"),
            );
        }
    }
}

#[test]
fn quaternion_180_z_rotation_matrix() {
    let ctx = Context::new();
    // 180° about z: q = k (w=0,x=0,y=0,z=1)
    let q = Quaternion::new(ctx.int(0), ctx.int(0), ctx.int(0), ctx.int(1));
    let r = q.to_rotation_matrix();
    // Expected: R = [[-1,0,0],[0,-1,0],[0,0,1]]
    assert_close(
        r.get(0, 0).eval().eval_f64().unwrap(),
        -1.0,
        1e-10,
        "R[0,0]",
    );
    assert_close(
        r.get(1, 1).eval().eval_f64().unwrap(),
        -1.0,
        1e-10,
        "R[1,1]",
    );
    assert_close(r.get(2, 2).eval().eval_f64().unwrap(), 1.0, 1e-10, "R[2,2]");
    assert_close(r.get(0, 1).eval().eval_f64().unwrap(), 0.0, 1e-10, "R[0,1]");
}

// ---------------------------------------------------------------------------
// 3g. Quaternion subs and eval
// ---------------------------------------------------------------------------

#[test]
fn quaternion_subs_and_eval() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let q = Quaternion::new(t.cos(), t.sin(), ctx.int(0), ctx.int(0));
    let q_at_0 = q.subs(&t, &ctx.int(0)).eval();
    let (w, x, _, _) = quat_to_f64(&q_at_0);
    assert_close(w, 1.0, 1e-12, "cos(0) = 1");
    assert_close(x, 0.0, 1e-12, "sin(0) = 0");
}

// ---------------------------------------------------------------------------
// 3h. Quaternion multiplication associativity
// ---------------------------------------------------------------------------

#[test]
fn quaternion_multiplication_associativity() {
    let ctx = Context::new();
    let a = Quaternion::new(ctx.int(1), ctx.int(2), ctx.int(3), ctx.int(4));
    let b = Quaternion::new(ctx.int(5), ctx.int(6), ctx.int(7), ctx.int(8));
    let c = Quaternion::new(ctx.int(9), ctx.int(10), ctx.int(11), ctx.int(12));

    // (a*b)*c should equal a*(b*c)
    let ab_c = a.mul(&b).mul(&c);
    let a_bc = a.mul(&b.mul(&c));

    let (w1, x1, y1, z1) = quat_to_f64(&ab_c.eval());
    let (w2, x2, y2, z2) = quat_to_f64(&a_bc.eval());

    assert_close(w1, w2, 1e-6, "associativity w");
    assert_close(x1, x2, 1e-6, "associativity x");
    assert_close(y1, y2, 1e-6, "associativity y");
    assert_close(z1, z2, 1e-6, "associativity z");
}

// ---------------------------------------------------------------------------
// 3i. Quaternion conjugate of product
// ---------------------------------------------------------------------------

#[test]
fn quaternion_conjugate_of_product() {
    let ctx = Context::new();
    let a = Quaternion::new(ctx.int(1), ctx.int(2), ctx.int(3), ctx.int(4));
    let b = Quaternion::new(ctx.int(5), ctx.int(6), ctx.int(7), ctx.int(8));

    // (a*b)* = b* * a*  (reversal rule)
    let ab_conj = a.mul(&b).conjugate();
    let b_conj_a_conj = b.conjugate().mul(&a.conjugate());

    let (w1, x1, y1, z1) = quat_to_f64(&ab_conj.eval());
    let (w2, x2, y2, z2) = quat_to_f64(&b_conj_a_conj.eval());

    assert_close(w1, w2, 1e-6, "conj product w");
    assert_close(x1, x2, 1e-6, "conj product x");
    assert_close(y1, y2, 1e-6, "conj product y");
    assert_close(z1, z2, 1e-6, "conj product z");
}

// ---------------------------------------------------------------------------
// 3j. Quaternion norm is multiplicative
// ---------------------------------------------------------------------------

#[test]
fn quaternion_norm_multiplicative() {
    let ctx = Context::new();
    let a = Quaternion::new(ctx.int(1), ctx.int(2), ctx.int(3), ctx.int(4));
    let b = Quaternion::new(ctx.int(5), ctx.int(6), ctx.int(7), ctx.int(8));

    let norm_a = a.norm().eval().eval_f64().unwrap();
    let norm_b = b.norm().eval().eval_f64().unwrap();
    let norm_ab = a.mul(&b).norm().eval().eval_f64().unwrap();

    assert_close(norm_ab, norm_a * norm_b, 1e-6, "|a*b| = |a|*|b|");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. VECTOR OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

use symplex::matrix::{cross, dot};
use symplex::vector::{curl, divergence, gradient, is_conservative, is_solenoidal, laplacian};

// ---------------------------------------------------------------------------
// 4a. Dot product
// ---------------------------------------------------------------------------

#[test]
fn vector_dot_product_basic() {
    let ctx = Context::new();
    let a = Matrix::col_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]).unwrap();
    let b = Matrix::col_vector(vec![ctx.int(4), ctx.int(5), ctx.int(6)]).unwrap();
    let d = dot(&a, &b).unwrap().eval();
    // 1*4 + 2*5 + 3*6 = 4+10+18 = 32
    assert_close(d.eval_f64().unwrap(), 32.0, 1e-12, "dot product");
}

#[test]
fn vector_dot_product_orthogonal() {
    let ctx = Context::new();
    let a = Matrix::col_vector(vec![ctx.int(1), ctx.int(0), ctx.int(0)]).unwrap();
    let b = Matrix::col_vector(vec![ctx.int(0), ctx.int(1), ctx.int(0)]).unwrap();
    let d = dot(&a, &b).unwrap().eval();
    assert_close(d.eval_f64().unwrap(), 0.0, 1e-12, "orthogonal dot = 0");
}

#[test]
fn vector_dot_product_self_is_norm_squared() {
    let ctx = Context::new();
    let a = Matrix::col_vector(vec![ctx.int(3), ctx.int(4)]).unwrap();
    let d = dot(&a, &a).unwrap().eval();
    assert_close(d.eval_f64().unwrap(), 25.0, 1e-12, "v·v = |v|²");
}

// ---------------------------------------------------------------------------
// 4b. Cross product
// ---------------------------------------------------------------------------

#[test]
fn vector_cross_product_basic() {
    let ctx = Context::new();
    let a = Matrix::col_vector(vec![ctx.int(1), ctx.int(0), ctx.int(0)]).unwrap();
    let b = Matrix::col_vector(vec![ctx.int(0), ctx.int(1), ctx.int(0)]).unwrap();
    let c = cross(&a, &b).unwrap();
    // i × j = k
    assert_close(c.get(0, 0).eval_f64().unwrap(), 0.0, 1e-12, "cross x");
    assert_close(c.get(1, 0).eval_f64().unwrap(), 0.0, 1e-12, "cross y");
    assert_close(c.get(2, 0).eval_f64().unwrap(), 1.0, 1e-12, "cross z");
}

#[test]
fn vector_cross_product_anticommutative() {
    let ctx = Context::new();
    let a = Matrix::col_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]).unwrap();
    let b = Matrix::col_vector(vec![ctx.int(4), ctx.int(5), ctx.int(6)]).unwrap();

    let ab = cross(&a, &b).unwrap();
    let ba = cross(&b, &a).unwrap();

    for i in 0..3 {
        let val_ab = ab.get(i, 0).eval().eval_f64().unwrap();
        let val_ba = ba.get(i, 0).eval().eval_f64().unwrap();
        assert_close(val_ab, -val_ba, 1e-12, &format!("a×b = -(b×a) [{i}]"));
    }
}

#[test]
fn vector_cross_product_self_is_zero() {
    let ctx = Context::new();
    let a = Matrix::col_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]).unwrap();
    let c = cross(&a, &a).unwrap();
    for i in 0..3 {
        assert_close(
            c.get(i, 0).eval().eval_f64().unwrap(),
            0.0,
            1e-12,
            &format!("a×a = 0 [{i}]"),
        );
    }
}

#[test]
fn vector_cross_product_triple_scalar() {
    let ctx = Context::new();
    // a · (b × c) = det([a; b; c])
    let a = Matrix::col_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]).unwrap();
    let b = Matrix::col_vector(vec![ctx.int(4), ctx.int(5), ctx.int(6)]).unwrap();
    let c = Matrix::col_vector(vec![ctx.int(7), ctx.int(8), ctx.int(10)]).unwrap();

    let bc = cross(&b, &c).unwrap();
    let triple = dot(&a, &bc).unwrap().eval().eval_f64().unwrap();

    // det of the matrix formed by a, b, c as rows
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
        vec![ctx.int(7), ctx.int(8), ctx.int(10)],
    ])
    .unwrap();
    let det_val = m.det().unwrap().eval().eval_f64().unwrap();

    assert_close(triple, det_val, 1e-10, "a·(b×c) = det");
}

// ---------------------------------------------------------------------------
// 4c. Gradient
// ---------------------------------------------------------------------------

#[test]
fn vector_gradient_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // f = x² + 2xy + y²
    let f = &x.powi(2) + &(&ctx.int(2) * &(&x * &y)) + &y.powi(2);
    let g = gradient(&f, &[&x, &y]).unwrap();
    assert_eq!(g.shape(), (2, 1));
    // ∂f/∂x = 2x + 2y
    // ∂f/∂y = 2x + 2y
    let gx = g
        .get(0, 0)
        .subs(&x, &ctx.int(1))
        .subs(&y, &ctx.int(1))
        .eval();
    assert_close(gx.eval_f64().unwrap(), 4.0, 1e-10, "∂f/∂x at (1,1)");
    let gy = g
        .get(1, 0)
        .subs(&x, &ctx.int(1))
        .subs(&y, &ctx.int(1))
        .eval();
    assert_close(gy.eval_f64().unwrap(), 4.0, 1e-10, "∂f/∂y at (1,1)");
}

// ---------------------------------------------------------------------------
// 4d. Divergence
// ---------------------------------------------------------------------------

#[test]
fn vector_divergence_basic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    // F = [x², y², z²]
    let field = Matrix::col_vector(vec![x.powi(2), y.powi(2), z.powi(2)]).unwrap();
    let div = divergence(&field, &[&x, &y, &z]).unwrap();
    // div F = 2x + 2y + 2z
    let val = div
        .subs(&x, &ctx.int(1))
        .subs(&y, &ctx.int(1))
        .subs(&z, &ctx.int(1))
        .eval();
    assert_close(val.eval_f64().unwrap(), 6.0, 1e-10, "div F at (1,1,1)");
}

// ---------------------------------------------------------------------------
// 4e. Curl
// ---------------------------------------------------------------------------

#[test]
fn vector_curl_of_gradient_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    // f = x²y + y²z + z²x
    let f = &(&x.powi(2) * &y) + &(&y.powi(2) * &z) + &(&z.powi(2) * &x);
    let g = gradient(&f, &[&x, &y, &z]).unwrap();
    // curl(grad(f)) should be zero
    let c = curl(&g, &[&x, &y, &z]).unwrap();
    for i in 0..3 {
        let val = c
            .get(i, 0)
            .eval()
            .simplify()
            .subs(&x, &ctx.int(1))
            .subs(&y, &ctx.int(1))
            .subs(&z, &ctx.int(1))
            .eval();
        assert_close(
            val.eval_f64().unwrap(),
            0.0,
            1e-10,
            &format!("curl(grad f).unwrap()[{i}]"),
        );
    }
}

// ---------------------------------------------------------------------------
// 4f. Laplacian
// ---------------------------------------------------------------------------

#[test]
fn vector_laplacian_quadratic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // f = x² + y²  →  ∇²f = 2 + 2 = 4
    let f = &x.powi(2) + &y.powi(2);
    let lap = laplacian(&f, &[&x, &y]).unwrap().eval();
    assert_close(lap.eval_f64().unwrap(), 4.0, 1e-10, "∇²(x²+y²)");
}

// ---------------------------------------------------------------------------
// 4g. Conservative / solenoidal tests
// ---------------------------------------------------------------------------

#[test]
fn vector_is_conservative_gradient_field() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    // F = grad(x²+y²+z²) = [2x, 2y, 2z] — conservative
    let f = &x.powi(2) + &y.powi(2) + &z.powi(2);
    let field = gradient(&f, &[&x, &y, &z]).unwrap();
    assert_eq!(
        is_conservative(&field, &[&x, &y, &z]),
        Some(true),
        "gradient field should be conservative"
    );
}

#[test]
fn vector_is_solenoidal_constant_field() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    // Constant field has zero divergence
    let field = Matrix::col_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]).unwrap();
    assert_eq!(
        is_solenoidal(&field, &[&x, &y, &z]),
        Some(true),
        "constant field should be solenoidal"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. CONTROL SYSTEMS
// ═══════════════════════════════════════════════════════════════════════════

use symplex::control::{TransferFunction, is_routh_stable, routh_array};

// ---------------------------------------------------------------------------
// 5a. Transfer function construction and evaluation
// ---------------------------------------------------------------------------

#[test]
fn control_tf_from_coeffs() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    // G(s) = 1 / (s² + 3s + 2)
    let g = TransferFunction::from_coeffs(&[1], &[2, 3, 1], &s);
    // DC gain: G(0) = 1/2
    let dc = g.dc_gain().eval();
    assert_close(dc.eval_f64().unwrap(), 0.5, 1e-10, "DC gain = 1/2");
}

#[test]
fn control_tf_eval_at() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    // G(s) = s / (s + 1)
    let g = TransferFunction::from_coeffs(&[0, 1], &[1, 1], &s);
    // G(1) = 1/2
    let val = g.eval_at(&ctx.int(1)).eval();
    assert_close(val.eval_f64().unwrap(), 0.5, 1e-10, "G(1) = 1/2");
    // G(0) = 0
    let dc = g.dc_gain().eval();
    assert_close(dc.eval_f64().unwrap(), 0.0, 1e-10, "G(0) = 0");
}

// ---------------------------------------------------------------------------
// 5b. Series and parallel connections
// ---------------------------------------------------------------------------

#[test]
fn control_tf_series_connection() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    // G1(s) = 1/(s+1), G2(s) = 1/(s+2)
    let g1 = TransferFunction::from_coeffs(&[1], &[1, 1], &s);
    let g2 = TransferFunction::from_coeffs(&[1], &[2, 1], &s);
    let gs = g1.series(&g2);
    // Series DC gain = G1(0)*G2(0) = 1 * 1/2 = 1/2
    let dc = gs.dc_gain().eval();
    assert_close(dc.eval_f64().unwrap(), 0.5, 1e-10, "series DC gain");
}

#[test]
fn control_tf_parallel_connection() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    // G1(s) = 1/(s+1), G2(s) = 1/(s+2)
    let g1 = TransferFunction::from_coeffs(&[1], &[1, 1], &s);
    let g2 = TransferFunction::from_coeffs(&[1], &[2, 1], &s);
    let gp = g1.parallel(&g2);
    // Parallel DC gain = G1(0)+G2(0) = 1 + 1/2 = 3/2
    let dc = gp.dc_gain().eval();
    assert_close(dc.eval_f64().unwrap(), 1.5, 1e-10, "parallel DC gain");
}

// ---------------------------------------------------------------------------
// 5c. Feedback
// ---------------------------------------------------------------------------

#[test]
fn control_tf_unity_feedback() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    // G(s) = 10/(s+1)
    let g = TransferFunction::from_coeffs(&[10], &[1, 1], &s);
    let gcl = g.feedback();
    // G_cl(0) = num(0) / (den(0)+num(0)) = 10 / (1+10) = 10/11
    let dc = gcl.dc_gain().eval();
    assert_close(
        dc.eval_f64().unwrap(),
        10.0 / 11.0,
        1e-10,
        "feedback DC gain",
    );
}

#[test]
fn control_tf_feedback_with_controller() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    // G(s) = 1/(s+1), H(s) = 2
    let g = TransferFunction::from_coeffs(&[1], &[1, 1], &s);
    let h = TransferFunction::from_coeffs(&[2], &[1], &s);
    let gcl = g.feedback_with(&h);
    // G_cl(0) = (1*1) / (1*1 + 1*2) = 1/3
    let dc = gcl.dc_gain().eval();
    assert_close(
        dc.eval_f64().unwrap(),
        1.0 / 3.0,
        1e-10,
        "feedback_with DC gain",
    );
}

// ---------------------------------------------------------------------------
// 5d. Poles and zeros
// ---------------------------------------------------------------------------

#[test]
fn control_tf_poles_simple() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    // G(s) = 1 / (s² + 3s + 2) = 1 / ((s+1)(s+2))
    let g = TransferFunction::from_coeffs(&[1], &[2, 3, 1], &s);
    let poles = g.poles();
    // Should find poles at -1 and -2
    assert!(poles.len() >= 2, "should find 2 poles, got {}", poles.len());
    let mut pole_vals: Vec<f64> = poles.iter().filter_map(|p| p.eval_f64().ok()).collect();
    pole_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if pole_vals.len() >= 2 {
        assert_close(pole_vals[0], -2.0, 1e-8, "pole at -2");
        assert_close(pole_vals[1], -1.0, 1e-8, "pole at -1");
    }
}

#[test]
fn control_tf_zeros_simple() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    // G(s) = (s+3) / (s² + 3s + 2) → numerator = 3 + s
    let g = TransferFunction::from_coeffs(&[3, 1], &[2, 3, 1], &s);
    let zeros = g.zeros();
    assert!(!zeros.is_empty(), "should find 1 zero");
    if !zeros.is_empty() {
        let zval = zeros[0].eval_f64().unwrap();
        assert_close(zval, -3.0, 1e-8, "zero at -3");
    }
}

// ---------------------------------------------------------------------------
// 5e. Routh-Hurwitz stability
// ---------------------------------------------------------------------------

#[test]
fn control_routh_stable_system() {
    let ctx = Context::new();
    // s² + 3s + 2 = (s+1)(s+2), both poles negative → stable
    let coeffs = vec![ctx.int(1), ctx.int(3), ctx.int(2)];
    let stable = is_routh_stable(&coeffs);
    assert_eq!(stable, Some(true), "s²+3s+2 should be stable");
}

/// BUG FOUND: `is_routh_stable` returns `None` instead of `Some(false)` for
/// `s² - 1 = (s+1)(s-1)` because the middle coefficient is zero, making the
/// Routh array's second-row pivot zero. `routh_array` divides by zero,
/// producing an unevaluable expression. `is_routh_stable` then fails
/// `eval_f64()` and returns `None` instead of detecting the zero first-column
/// entry and returning `Some(false)`.
///
/// The classical Routh-Hurwitz algorithm handles this via the "epsilon method"
/// (replace the zero pivot with a small ε, then take the limit). The current
/// implementation lacks this handling.
#[test]
fn control_routh_unstable_system_zero_pivot_bug() {
    let ctx = Context::new();
    // s² - 1 = (s+1)(s-1), has positive root → unstable
    let coeffs = vec![ctx.int(1), ctx.int(0), ctx.int(-1)];
    let stable = is_routh_stable(&coeffs);
    assert_eq!(
        stable,
        Some(false),
        "s²-1 has a positive real root and should be unstable"
    );
}

#[test]
fn control_routh_unstable_system_nonzero_pivot() {
    let ctx = Context::new();
    // s² + s - 2 = (s+2)(s-1), has positive root → unstable, no zero pivot
    let coeffs = vec![ctx.int(1), ctx.int(1), ctx.int(-2)];
    let stable = is_routh_stable(&coeffs);
    assert_eq!(stable, Some(false), "s²+s-2 should be unstable");
}

#[test]
fn control_routh_array_size() {
    let ctx = Context::new();
    // s³ + 2s² + 3s + 4 → 4 coefficients → 4 rows
    let coeffs = vec![ctx.int(1), ctx.int(2), ctx.int(3), ctx.int(4)];
    let table = routh_array(&coeffs).unwrap();
    assert_eq!(table.len(), 4, "Routh table should have 4 rows");
}

// ---------------------------------------------------------------------------
// 5f. State-space model
// ---------------------------------------------------------------------------

#[test]
fn control_state_space_dimensions() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(0), ctx.int(1)],
        vec![ctx.int(-2), ctx.int(-3)],
    ])
    .unwrap();
    let b = Matrix::new(vec![vec![ctx.int(0)], vec![ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![ctx.int(1), ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();

    let ss = StateSpace::new(a, b, c, d).unwrap();
    assert_eq!(ss.num_states(), 2);
    assert_eq!(ss.num_inputs(), 1);
    assert_eq!(ss.num_outputs(), 1);
}

#[test]
fn control_state_space_controllability() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(0), ctx.int(1)],
        vec![ctx.int(-2), ctx.int(-3)],
    ])
    .unwrap();
    let b = Matrix::new(vec![vec![ctx.int(0)], vec![ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![ctx.int(1), ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();

    let ss = StateSpace::new(a, b, c, d).unwrap();
    // System with A = [[0,1],[-2,-3]], B = [[0],[1]]
    // Controllability matrix = [B, AB] = [[0, 1], [1, -3]]
    // det = -1 ≠ 0 → controllable
    assert!(ss.is_controllable(), "system should be controllable");
}

#[test]
fn control_state_space_observability() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(0), ctx.int(1)],
        vec![ctx.int(-2), ctx.int(-3)],
    ])
    .unwrap();
    let b = Matrix::new(vec![vec![ctx.int(0)], vec![ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![ctx.int(1), ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();

    let ss = StateSpace::new(a, b, c, d).unwrap();
    // Observability matrix = [C; CA] = [[1,0],[0,1]] → rank 2 → observable
    assert!(ss.is_observable(), "system should be observable");
}

#[test]
fn control_state_space_stability() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(0), ctx.int(1)],
        vec![ctx.int(-2), ctx.int(-3)],
    ])
    .unwrap();
    let b = Matrix::new(vec![vec![ctx.int(0)], vec![ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![ctx.int(1), ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();

    let ss = StateSpace::new(a, b, c, d).unwrap();
    // Eigenvalues of A: λ² + 3λ + 2 = 0 → λ = -1, -2 → stable
    let stable = ss.is_stable();
    assert_eq!(stable, Some(true), "system should be stable");
}

#[test]
fn control_state_space_char_poly() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(0), ctx.int(1)],
        vec![ctx.int(-2), ctx.int(-3)],
    ])
    .unwrap();
    let b = Matrix::new(vec![vec![ctx.int(0)], vec![ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![ctx.int(1), ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d).unwrap();

    let s = ctx.symbol("s");
    let cp = ss.char_poly(&s).eval().expand();
    // Should be s² + 3s + 2
    let at_0 = cp.subs(&s, &ctx.int(0)).eval();
    assert_close(at_0.eval_f64().unwrap(), 2.0, 1e-10, "char_poly(0) = 2");
    let at_1 = cp.subs(&s, &ctx.int(1)).eval();
    assert_close(at_1.eval_f64().unwrap(), 6.0, 1e-10, "char_poly(1) = 6");
}

// ---------------------------------------------------------------------------
// 5g. Discretization (ZOH)
// ---------------------------------------------------------------------------

#[test]
fn control_state_space_discretize() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(0), ctx.int(1)],
        vec![ctx.int(-2), ctx.int(-3)],
    ])
    .unwrap();
    let b = Matrix::new(vec![vec![ctx.int(0)], vec![ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![ctx.int(1), ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d).unwrap();

    let dt = ctx.rational(1, 10); // dt = 0.1
    let sd = ss.discretize_zoh(&dt, 10).unwrap();
    // Discrete system should have same number of states
    assert_eq!(sd.num_states(), 2);
    assert_eq!(sd.num_inputs(), 1);
    assert_eq!(sd.num_outputs(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. ROBOTICS
// ═══════════════════════════════════════════════════════════════════════════

use symplex::robotics::{
    DhLink, EulerConvention, dh_matrix, fk_chain, fk_position, fk_rotation, homogeneous, rot_euler,
    rot_x, rot_y, rot_z, skew3, translation,
};

// ---------------------------------------------------------------------------
// 6a. DH matrix structure
// ---------------------------------------------------------------------------

#[test]
fn robotics_dh_matrix_shape() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let t = dh_matrix(&zero, &zero, &zero, &zero);
    assert_eq!(t.shape(), (4, 4));
}

#[test]
fn robotics_dh_identity_when_all_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let t = dh_matrix(&zero, &zero, &zero, &zero);
    // With all zeros: cos(0)=1, sin(0)=0
    // T = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let val = t.get(i, j).eval().eval_f64().unwrap();
            assert_close(val, expected, 1e-10, &format!("DH zero [{i},{j}]"));
        }
    }
}

#[test]
fn robotics_dh_bottom_row() {
    let ctx = Context::new();
    let theta = ctx.symbol("theta");
    let d = ctx.symbol("d");
    let a = ctx.symbol("a");
    let alpha = ctx.symbol("alpha");
    let t = dh_matrix(&theta, &d, &a, &alpha);
    // Bottom row should always be [0, 0, 0, 1]
    assert_close(t.get(3, 0).eval_f64().unwrap(), 0.0, 1e-12, "T[3,0]=0");
    assert_close(t.get(3, 1).eval_f64().unwrap(), 0.0, 1e-12, "T[3,1]=0");
    assert_close(t.get(3, 2).eval_f64().unwrap(), 0.0, 1e-12, "T[3,2]=0");
    assert_close(t.get(3, 3).eval_f64().unwrap(), 1.0, 1e-12, "T[3,3]=1");
}

// ---------------------------------------------------------------------------
// 6b. FK chain
// ---------------------------------------------------------------------------

#[test]
fn robotics_fk_chain_single_joint() {
    let ctx = Context::new();
    let theta = ctx.symbol("theta");
    let zero = ctx.int(0);
    let l = ctx.symbol("L");

    let params = [DhLink {
        theta: &theta,
        d: &zero,
        a: &l,
        alpha: &zero,
    }];
    let t = fk_chain(&params);
    assert_eq!(t.shape(), (4, 4));
}

#[test]
fn robotics_fk_position_1dof_at_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let l = ctx.int(1);

    // Single joint at θ=0, a=1
    let (px, py, pz) = fk_position(&[DhLink {
        theta: &zero,
        d: &zero,
        a: &l,
        alpha: &zero,
    }]);
    assert_close(px.eval_f64().unwrap(), 1.0, 1e-10, "fk x at θ=0");
    assert_close(py.eval_f64().unwrap(), 0.0, 1e-10, "fk y at θ=0");
    assert_close(pz.eval_f64().unwrap(), 0.0, 1e-10, "fk z at θ=0");
}

#[test]
fn robotics_fk_position_1dof_at_90deg() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let half_pi = &pi / 2;
    let zero = ctx.int(0);
    let l = ctx.int(1);

    let (px, py, pz) = fk_position(&[DhLink {
        theta: &half_pi,
        d: &zero,
        a: &l,
        alpha: &zero,
    }]);
    assert_close(px.eval_f64().unwrap(), 0.0, 1e-10, "fk x at θ=90°");
    assert_close(py.eval_f64().unwrap(), 1.0, 1e-10, "fk y at θ=90°");
    assert_close(pz.eval_f64().unwrap(), 0.0, 1e-10, "fk z at θ=90°");
}

#[test]
fn robotics_fk_rotation_shape() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let l = ctx.symbol("L");
    let theta = ctx.symbol("theta");
    let r = fk_rotation(&[DhLink {
        theta: &theta,
        d: &zero,
        a: &l,
        alpha: &zero,
    }]);
    assert_eq!(r.shape(), (3, 3));
}

// ---------------------------------------------------------------------------
// 6c. Rotation matrices
// ---------------------------------------------------------------------------

#[test]
fn robotics_rot_x_shape_and_identity_at_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let r = rot_x(&zero);
    assert_eq!(r.shape(), (3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(
                r.get(i, j).eval().eval_f64().unwrap(),
                expected,
                1e-10,
                &format!("Rx(0)[{i},{j}]"),
            );
        }
    }
}

#[test]
fn robotics_rot_y_at_zero_is_identity() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let r = rot_y(&zero);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(
                r.get(i, j).eval().eval_f64().unwrap(),
                expected,
                1e-10,
                &format!("Ry(0)[{i},{j}]"),
            );
        }
    }
}

#[test]
fn robotics_rot_z_at_zero_is_identity() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let r = rot_z(&zero);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(
                r.get(i, j).eval().eval_f64().unwrap(),
                expected,
                1e-10,
                &format!("Rz(0)[{i},{j}]"),
            );
        }
    }
}

#[test]
fn robotics_rotation_matrix_is_orthogonal() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let angle = &pi / 6; // 30°
    let r = rot_z(&angle);
    // R * Rᵀ should be I
    let rt = r.transpose();
    let product = r.matmul(&rt).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(
                product.get(i, j).eval().eval_f64().unwrap(),
                expected,
                1e-10,
                &format!("RRᵀ[{i},{j}]"),
            );
        }
    }
}

#[test]
fn robotics_rotation_matrix_det_is_one() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let angle = &pi / 4; // 45°
    let r = rot_x(&angle);
    let d = r.det().unwrap().eval();
    assert_close(d.eval_f64().unwrap(), 1.0, 1e-10, "det(Rx) = 1");
}

// ---------------------------------------------------------------------------
// 6d. Euler angles
// ---------------------------------------------------------------------------

#[test]
fn robotics_euler_zyx_identity_at_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let r = rot_euler(&zero, &zero, &zero, EulerConvention::ZYX);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(
                r.get(i, j).eval().eval_f64().unwrap(),
                expected,
                1e-10,
                &format!("Euler ZYX(0,0,0)[{i},{j}]"),
            );
        }
    }
}

#[test]
fn robotics_euler_xyz_identity_at_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let r = rot_euler(&zero, &zero, &zero, EulerConvention::XYZ);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(
                r.get(i, j).eval().eval_f64().unwrap(),
                expected,
                1e-10,
                &format!("Euler XYZ(0,0,0)[{i},{j}]"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 6e. Skew-symmetric matrix
// ---------------------------------------------------------------------------

#[test]
fn robotics_skew3_is_antisymmetric() {
    let ctx = Context::new();
    let a = ctx.int(1);
    let b = ctx.int(2);
    let c = ctx.int(3);
    let s = skew3(&a, &b, &c);
    assert_eq!(s.shape(), (3, 3));
    // Check s + sᵀ = 0
    let st = s.transpose();
    let sum = s.add(&st).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert_close(
                sum.get(i, j).eval().eval_f64().unwrap(),
                0.0,
                1e-12,
                &format!("skew + skewᵀ [{i},{j}]"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 6f. Homogeneous transformation
// ---------------------------------------------------------------------------

#[test]
fn robotics_homogeneous_from_identity_rotation() {
    let ctx = Context::new();
    let r = Matrix::identity(&ctx, 3).unwrap();
    let pos = [ctx.int(1), ctx.int(2), ctx.int(3)];
    let t = homogeneous(&r, &pos).unwrap();
    assert_eq!(t.shape(), (4, 4));
    // Check translation column
    assert_close(t.get(0, 3).eval_f64().unwrap(), 1.0, 1e-12, "T[0,3] = 1");
    assert_close(t.get(1, 3).eval_f64().unwrap(), 2.0, 1e-12, "T[1,3] = 2");
    assert_close(t.get(2, 3).eval_f64().unwrap(), 3.0, 1e-12, "T[2,3] = 3");
    // Check bottom row
    assert_close(t.get(3, 3).eval_f64().unwrap(), 1.0, 1e-12, "T[3,3] = 1");
}

#[test]
fn robotics_translation_matrix() {
    let ctx = Context::new();
    let x = ctx.int(5);
    let y = ctx.int(10);
    let z = ctx.int(15);
    let t = translation(&x, &y, &z);
    assert_eq!(t.shape(), (4, 4));
    assert_close(t.get(0, 3).eval_f64().unwrap(), 5.0, 1e-12, "tx");
    assert_close(t.get(1, 3).eval_f64().unwrap(), 10.0, 1e-12, "ty");
    assert_close(t.get(2, 3).eval_f64().unwrap(), 15.0, 1e-12, "tz");
    // Upper-left 3×3 should be identity
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(
                t.get(i, j).eval_f64().unwrap(),
                expected,
                1e-12,
                &format!("trans rotation [{i},{j}]"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 6g. Typed DH API
// ---------------------------------------------------------------------------

#[test]
fn robotics_fk_position_typed() {
    use symplex::robotics::{DhParams, fk_position_typed};
    use symplex::units::si::*;

    let ctx = Context::new();
    let theta1 = Angle::from_ex(ctx.int(0)); // θ=0
    let l1 = Length::constant(&ctx, 1);
    let zero_l = Length::zero(&ctx);
    let zero_a = Angle::zero(&ctx);

    let (px, py, pz) = fk_position_typed(&[DhParams {
        theta: &theta1,
        d: &zero_l,
        a: &l1,
        alpha: &zero_a,
    }]);
    assert_close(px.eval_f64().unwrap(), 1.0, 1e-10, "typed fk px");
    assert_close(py.eval_f64().unwrap(), 0.0, 1e-10, "typed fk py");
    assert_close(pz.eval_f64().unwrap(), 0.0, 1e-10, "typed fk pz");
}

// ---------------------------------------------------------------------------
// 6h. Two-link FK chain
// ---------------------------------------------------------------------------

#[test]
fn robotics_two_link_fk_position() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let pi = ctx.pi();
    let half_pi = &pi / 2;
    let l1 = ctx.int(1);
    let l2 = ctx.int(1);

    // Joint 1 at 0°, Joint 2 at 90°
    let (px, py, pz) = fk_position(&[
        DhLink {
            theta: &zero,
            d: &zero,
            a: &l1,
            alpha: &zero,
        },
        DhLink {
            theta: &half_pi,
            d: &zero,
            a: &l2,
            alpha: &zero,
        },
    ]);
    // Link 1 extends along x by 1, Link 2 rotated 90° from that → adds 1 in y
    // But note: in planar case with no alpha, the second link extends along
    // the direction θ₁+θ₂ from the first joint coordinate system
    // At θ₁=0, θ₂=π/2: x = l1*cos(0) + l2*cos(π/2) = 1+0 = 1
    //                    y = l1*sin(0) + l2*sin(π/2) = 0+1 = 1
    assert_close(px.eval_f64().unwrap(), 1.0, 1e-10, "2-link x");
    assert_close(py.eval_f64().unwrap(), 1.0, 1e-10, "2-link y");
    assert_close(pz.eval_f64().unwrap(), 0.0, 1e-10, "2-link z");
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. COMBINATORICS EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

use num_bigint::BigInt;
use symplex::combinatorics::{multinomial, partition_count, stirling1, stirling2};

// ---------------------------------------------------------------------------
// 7a. Stirling2 edge cases
// ---------------------------------------------------------------------------

#[test]
fn combinatorics_stirling2_zero_zero() {
    assert_eq!(stirling2(0, 0), Some(BigInt::from(1)));
}

#[test]
fn combinatorics_stirling2_n_zero_is_zero() {
    for n in 1..=10 {
        assert_eq!(
            stirling2(n, 0),
            Some(BigInt::from(0)),
            "S({n}, 0) should be 0"
        );
    }
}

#[test]
fn combinatorics_stirling2_n_n_is_one() {
    for n in 0..=20 {
        assert_eq!(
            stirling2(n, n),
            Some(BigInt::from(1)),
            "S({n}, {n}) should be 1"
        );
    }
}

#[test]
fn combinatorics_stirling2_n_1_is_one() {
    for n in 1..=20 {
        assert_eq!(
            stirling2(n, 1),
            Some(BigInt::from(1)),
            "S({n}, 1) should be 1"
        );
    }
}

#[test]
fn combinatorics_stirling2_k_gt_n_is_zero() {
    assert_eq!(stirling2(3, 5), Some(BigInt::from(0)));
    assert_eq!(stirling2(0, 1), Some(BigInt::from(0)));
    assert_eq!(stirling2(5, 10), Some(BigInt::from(0)));
}

#[test]
fn combinatorics_stirling2_negative_is_zero() {
    assert_eq!(stirling2(-1, 0), Some(BigInt::from(0)));
    assert_eq!(stirling2(5, -1), Some(BigInt::from(0)));
    assert_eq!(stirling2(-3, -2), Some(BigInt::from(0)));
}

#[test]
fn combinatorics_stirling2_known_values() {
    // S(4, 2) = 7
    assert_eq!(stirling2(4, 2), Some(BigInt::from(7)));
    // S(5, 3) = 25
    assert_eq!(stirling2(5, 3), Some(BigInt::from(25)));
    // S(6, 3) = 90
    assert_eq!(stirling2(6, 3), Some(BigInt::from(90)));
    // S(7, 4) = 350
    assert_eq!(stirling2(7, 4), Some(BigInt::from(350)));
    // S(8, 5) = 1050  (verified via recurrence)
    assert_eq!(stirling2(8, 5), Some(BigInt::from(1050)));
    // S(10, 3) = 9330
    assert_eq!(stirling2(10, 3), Some(BigInt::from(9330)));
}

#[test]
fn combinatorics_stirling2_n_2_is_2_pow_nm1_minus_1() {
    // S(n, 2) = 2^(n-1) - 1
    for n in 2..=15 {
        let expected = (1i64 << (n - 1)) - 1;
        assert_eq!(
            stirling2(n, 2),
            Some(BigInt::from(expected)),
            "S({n}, 2) = 2^({}) - 1 = {expected}",
            n - 1
        );
    }
}

#[test]
fn combinatorics_stirling2_n_nm1_is_binomial() {
    // S(n, n-1) = C(n, 2) = n*(n-1)/2
    for n in 2..=15 {
        let expected = n * (n - 1) / 2;
        assert_eq!(
            stirling2(n, n - 1),
            Some(BigInt::from(expected)),
            "S({n}, {}) = C({n},2) = {expected}",
            n - 1
        );
    }
}

#[test]
fn combinatorics_stirling2_recurrence() {
    // S(n, k) = k * S(n-1, k) + S(n-1, k-1)
    for n in 2..=10 {
        for k in 1..=n {
            let lhs = stirling2(n, k).unwrap();
            let t1 = BigInt::from(k) * stirling2(n - 1, k).unwrap();
            let t2 = stirling2(n - 1, k - 1).unwrap();
            let rhs = t1 + t2;
            assert_eq!(lhs, rhs, "Recurrence failed for S({n}, {k})");
        }
    }
}

// Bell numbers: B(n) = sum_{k=0}^{n} S(n, k)
#[test]
fn combinatorics_stirling2_bell_numbers() {
    let bell: Vec<i64> = vec![1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975];
    for (n, &expected) in bell.iter().enumerate() {
        let mut sum = BigInt::from(0);
        for k in 0..=n {
            sum += stirling2(n as i64, k as i64).unwrap();
        }
        assert_eq!(
            sum,
            BigInt::from(expected),
            "B({n}) = sum S({n},k) = {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// 7b. Stirling1 (signed) edge cases
// ---------------------------------------------------------------------------

#[test]
fn combinatorics_stirling1_zero_zero() {
    assert_eq!(stirling1(0, 0), Some(BigInt::from(1)));
}

#[test]
fn combinatorics_stirling1_n_zero_is_zero() {
    for n in 1..=10 {
        assert_eq!(
            stirling1(n, 0),
            Some(BigInt::from(0)),
            "s({n}, 0) should be 0"
        );
    }
}

#[test]
fn combinatorics_stirling1_n_n_is_one() {
    for n in 0..=20 {
        assert_eq!(
            stirling1(n, n),
            Some(BigInt::from(1)),
            "s({n}, {n}) should be 1"
        );
    }
}

#[test]
fn combinatorics_stirling1_k_gt_n_is_zero() {
    assert_eq!(stirling1(3, 5), Some(BigInt::from(0)));
}

#[test]
fn combinatorics_stirling1_negative_is_zero() {
    assert_eq!(stirling1(-1, 0), Some(BigInt::from(0)));
    assert_eq!(stirling1(5, -1), Some(BigInt::from(0)));
}

#[test]
fn combinatorics_stirling1_known_values() {
    // s(3, 1) = 2
    assert_eq!(stirling1(3, 1), Some(BigInt::from(2)));
    // s(4, 1) = -6
    assert_eq!(stirling1(4, 1), Some(BigInt::from(-6)));
    // s(4, 2) = 11
    assert_eq!(stirling1(4, 2), Some(BigInt::from(11)));
    // s(4, 3) = -6
    assert_eq!(stirling1(4, 3), Some(BigInt::from(-6)));
    // s(5, 1) = 24
    assert_eq!(stirling1(5, 1), Some(BigInt::from(24)));
    // s(5, 2) = -50
    assert_eq!(stirling1(5, 2), Some(BigInt::from(-50)));
}

#[test]
fn combinatorics_stirling1_n_nm1() {
    // s(n, n-1) = -C(n, 2) = -n(n-1)/2
    for n in 2..=15i64 {
        let expected = -(n * (n - 1) / 2);
        assert_eq!(
            stirling1(n, n - 1),
            Some(BigInt::from(expected)),
            "s({n}, {}) = {expected}",
            n - 1
        );
    }
}

#[test]
fn combinatorics_stirling1_recurrence() {
    // s(n, k) = -(n-1) * s(n-1, k) + s(n-1, k-1)
    for n in 2..=10 {
        for k in 1..=n {
            let lhs = stirling1(n, k).unwrap();
            let t1 = BigInt::from(-(n - 1) as i64) * stirling1(n - 1, k).unwrap();
            let t2 = stirling1(n - 1, k - 1).unwrap();
            let rhs = t1 + t2;
            assert_eq!(lhs, rhs, "Recurrence failed for s({n}, {k})");
        }
    }
}

#[test]
fn combinatorics_stirling1_row_sum_zero_for_n_ge_2() {
    // sum_{k=0}^{n} s(n, k) = 0 for n >= 2 (because x^{(n)} at x=1 = 0 for n >= 2)
    // Actually: sum s(n,k) * 1^k = 1^{(n)} = 1*(1-1)*(1-2)*...*(1-(n-1))
    // = 0 for n >= 2
    for n in 2..=10 {
        let mut sum = BigInt::from(0);
        for k in 0..=n {
            sum += stirling1(n as i64, k as i64).unwrap();
        }
        assert_eq!(sum, BigInt::from(0), "sum s({n}, k) = 0 for n={n}");
    }
}

#[test]
fn combinatorics_stirling1_unsigned_row_sum_is_n_factorial() {
    // sum_{k=0}^{n} |s(n, k)| = n!
    // |s(n,k)| = (-1)^(n-k) * s(n,k) for signed Stirling
    for n in 0..=8 {
        let mut sum = BigInt::from(0);
        for k in 0..=n {
            let val = stirling1(n as i64, k as i64).unwrap();
            // Unsigned: (-1)^(n-k) * val
            let unsigned = if (n - k) % 2 == 0 { val } else { -val };
            sum += unsigned;
        }
        let fact: i64 = (1..=n as i64).product::<i64>().max(1);
        assert_eq!(sum, BigInt::from(fact), "sum |s({n}, k)| = {n}! = {fact}");
    }
}

// ---------------------------------------------------------------------------
// 7c. Stirling orthogonality
// ---------------------------------------------------------------------------

#[test]
fn combinatorics_stirling_orthogonality() {
    // sum_{j=k}^{n} s(n, j) * S(j, k) = delta(n, k)
    for n in 0..=6 {
        for k in 0..=n {
            let mut sum = BigInt::from(0);
            for j in k..=n {
                let s1 = stirling1(n as i64, j as i64).unwrap();
                let s2 = stirling2(j as i64, k as i64).unwrap();
                sum += s1 * s2;
            }
            let expected = if n == k {
                BigInt::from(1)
            } else {
                BigInt::from(0)
            };
            assert_eq!(
                sum, expected,
                "Orthogonality: sum s({n},j)*S(j,{k}) should be δ({n},{k})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 7d. Multinomial coefficients
// ---------------------------------------------------------------------------

#[test]
fn combinatorics_multinomial_basic() {
    // 6! / (2! 3! 1!) = 60
    assert_eq!(multinomial(6, &[2, 3, 1]), Some(BigInt::from(60)));
}

#[test]
fn combinatorics_multinomial_reduces_to_binomial() {
    // C(10, 3) = 10! / (3! 7!) = 120
    assert_eq!(multinomial(10, &[3, 7]), Some(BigInt::from(120)));
    // C(20, 10) = 184756
    assert_eq!(multinomial(20, &[10, 10]), Some(BigInt::from(184756)));
}

#[test]
fn combinatorics_multinomial_all_ones_is_factorial() {
    // n! / (1! * 1! * ... * 1!) = n!
    assert_eq!(multinomial(5, &[1, 1, 1, 1, 1]), Some(BigInt::from(120)));
    assert_eq!(multinomial(6, &[1, 1, 1, 1, 1, 1]), Some(BigInt::from(720)));
}

#[test]
fn combinatorics_multinomial_single_group() {
    // n! / n! = 1
    assert_eq!(multinomial(5, &[5]), Some(BigInt::from(1)));
    assert_eq!(multinomial(100, &[100]), Some(BigInt::from(1)));
}

#[test]
fn combinatorics_multinomial_with_zeros() {
    assert_eq!(multinomial(5, &[5, 0, 0]), Some(BigInt::from(1)));
    assert_eq!(multinomial(3, &[0, 3, 0]), Some(BigInt::from(1)));
}

#[test]
fn combinatorics_multinomial_sum_mismatch_returns_zero() {
    // If k's don't sum to n, result is 0
    assert_eq!(multinomial(6, &[2, 2]), Some(BigInt::from(0)));
    assert_eq!(multinomial(10, &[3, 3, 3]), Some(BigInt::from(0)));
}

#[test]
fn combinatorics_multinomial_negative_k_returns_zero() {
    assert_eq!(multinomial(5, &[3, -1, 3]), Some(BigInt::from(0)));
}

#[test]
fn combinatorics_multinomial_negative_n() {
    assert_eq!(multinomial(-1, &[1, 2]), Some(BigInt::from(0)));
}

#[test]
fn combinatorics_multinomial_zero_n_zero_ks() {
    // 0! / (0! * 0!) = 1 if sum = 0
    assert_eq!(multinomial(0, &[0, 0]), Some(BigInt::from(1)));
    assert_eq!(multinomial(0, &[0]), Some(BigInt::from(1)));
}

#[test]
fn combinatorics_multinomial_larger_values() {
    // 12! / (4! 4! 4!) = 34650
    assert_eq!(multinomial(12, &[4, 4, 4]), Some(BigInt::from(34650)));
    // 10! / (2! 3! 5!) = 2520
    assert_eq!(multinomial(10, &[2, 3, 5]), Some(BigInt::from(2520)));
}

// ---------------------------------------------------------------------------
// 7e. Partition counting
// ---------------------------------------------------------------------------

#[test]
fn combinatorics_partition_count_small() {
    // p(0) = 1 (empty partition)
    assert_eq!(partition_count(0), Some(BigInt::from(1)));
    // p(1) = 1
    assert_eq!(partition_count(1), Some(BigInt::from(1)));
    // p(2) = 2
    assert_eq!(partition_count(2), Some(BigInt::from(2)));
    // p(3) = 3
    assert_eq!(partition_count(3), Some(BigInt::from(3)));
    // p(4) = 5
    assert_eq!(partition_count(4), Some(BigInt::from(5)));
    // p(5) = 7
    assert_eq!(partition_count(5), Some(BigInt::from(7)));
}

#[test]
fn combinatorics_partition_count_medium() {
    // p(10) = 42
    assert_eq!(partition_count(10), Some(BigInt::from(42)));
    // p(20) = 627
    assert_eq!(partition_count(20), Some(BigInt::from(627)));
    // p(50) = 204226
    assert_eq!(partition_count(50), Some(BigInt::from(204226)));
}

#[test]
fn combinatorics_partition_count_100() {
    // p(100) = 190569292
    assert_eq!(partition_count(100), Some(BigInt::from(190569292i64)));
}

#[test]
fn combinatorics_partition_count_negative() {
    assert_eq!(partition_count(-1), Some(BigInt::from(0)));
    assert_eq!(partition_count(-100), Some(BigInt::from(0)));
}

// ---------------------------------------------------------------------------
// 7f. Bell numbers for larger n (via Stirling2 row sums)
// ---------------------------------------------------------------------------

#[test]
fn combinatorics_bell_numbers_larger() {
    // B(11) = 678570
    let mut b11 = BigInt::from(0);
    for k in 0..=11 {
        b11 += stirling2(11, k).unwrap();
    }
    assert_eq!(b11, BigInt::from(678570));

    // B(12) = 4213597
    let mut b12 = BigInt::from(0);
    for k in 0..=12 {
        b12 += stirling2(12, k).unwrap();
    }
    assert_eq!(b12, BigInt::from(4213597));

    // B(15) = 1382958545
    let mut b15 = BigInt::from(0);
    for k in 0..=15 {
        b15 += stirling2(15, k).unwrap();
    }
    assert_eq!(b15, BigInt::from(1382958545i64));
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. CROSS-DOMAIN: matrix + quaternion consistency
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quaternion_rotation_matches_rot_z() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let half_pi = &pi / 2;

    // Quaternion for 90° rotation about z
    let q = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(0), &ctx.int(1), &half_pi);
    let r_quat = q.to_rotation_matrix();

    // Rotation matrix directly
    let r_mat = rot_z(&half_pi);

    // They should match
    for i in 0..3 {
        for j in 0..3 {
            let qval = r_quat.get(i, j).eval().eval_f64().unwrap();
            let mval = r_mat.get(i, j).eval().eval_f64().unwrap();
            assert_close(qval, mval, 1e-8, &format!("quat vs rot_z [{i},{j}]"));
        }
    }
}

#[test]
fn quaternion_rotation_preserves_vector_norm() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let angle = &pi / 3; // 60°

    // Rotate a vector [1, 2, 3] by 60° about z
    let q = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(0), &ctx.int(1), &angle);
    let v = Quaternion::from_vector(&ctx.int(1), &ctx.int(2), &ctx.int(3));

    // q * v * q^{-1}
    let rotated = q.mul(&v).mul(&q.conjugate());
    let rotated_eval = rotated.eval();

    // Norm of vector part should be preserved: sqrt(1+4+9) = sqrt(14)
    let orig_norm = (1.0 + 4.0 + 9.0_f64).sqrt();
    let (_, rx, ry, rz) = quat_to_f64(&rotated_eval);
    let rot_norm = (rx * rx + ry * ry + rz * rz).sqrt();
    assert_close(rot_norm, orig_norm, 1e-8, "rotation preserves norm");
}
