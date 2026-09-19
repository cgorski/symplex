//! Integration tests for the Lagrangian dynamics module.
//!
//! Tests cover: total_time_derivative, euler_lagrange, mass_matrix,
//! christoffel_symbols, coriolis_matrix, gravity_vector, and
//! manipulator_equation.

use symplex::dynamics::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: check numeric equality with tolerance
// ═══════════════════════════════════════════════════════════════════════════

use symplex::prelude::*;
fn assert_near(val: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (val - expected).abs() < tol,
        "{msg}: got {val}, expected {expected} (diff = {})",
        (val - expected).abs()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. total_time_derivative tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn total_time_derivative_constant() {
    let ctx = Context::new();
    // d/dt of a constant = 0
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");
    let qdd = ctx.symbol("qdd");

    let c = ctx.int(5);
    let result = total_time_derivative(&c, &[(&q, &qd)], &[&qdd]).unwrap();
    let val = result.eval().eval_f64().unwrap();
    assert_near(val, 0.0, 1e-12, "d/dt(5) should be 0");
}

#[test]
fn total_time_derivative_linear_q() {
    let ctx = Context::new();
    // d/dt(q) = q_dot
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");
    let qdd = ctx.symbol("qdd");

    let result = total_time_derivative(&q, &[(&q, &qd)], &[&qdd]).unwrap();
    // Substitute qd = 7 and check
    let val = result.subs(&qd, &ctx.int(7)).eval().eval_f64().unwrap();
    assert_near(val, 7.0, 1e-12, "d/dt(q) should be qd");
}

#[test]
fn total_time_derivative_q_squared() {
    let ctx = Context::new();
    // d/dt(q²) = 2q·q_dot
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");
    let qdd = ctx.symbol("qdd");

    let q_sq = q.powi(2);
    let result = total_time_derivative(&q_sq, &[(&q, &qd)], &[&qdd]).unwrap();
    // Substitute q=3, qd=2: expect 2*3*2 = 12
    let val = result
        .subs(&q, &ctx.int(3))
        .subs(&qd, &ctx.int(2))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(val, 12.0, 1e-12, "d/dt(q^2) = 2*q*qd at q=3, qd=2");
}

#[test]
fn total_time_derivative_kinetic_energy() {
    let ctx = Context::new();
    // d/dt(½m·q̇²) = m·q̇·q̈
    let m = ctx.symbol("m");
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");
    let qdd = ctx.symbol("qdd");

    let half = ctx.rational(1, 2);
    let ke = &half * &m * &qd.powi(2);
    let result = total_time_derivative(&ke, &[(&q, &qd)], &[&qdd]).unwrap();

    // Substitute m=2, qd=3, qdd=5: expect 2*3*5 = 30
    let val = result
        .subs(&m, &ctx.int(2))
        .subs(&qd, &ctx.int(3))
        .subs(&qdd, &ctx.int(5))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(val, 30.0, 1e-12, "d/dt(½m·qd²) = m·qd·qdd");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. euler_lagrange tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn euler_lagrange_free_particle() {
    let ctx = Context::new();
    // Free particle: T = ½m·q̇², V = 0
    // Equation of motion: m·q̈ = τ
    let m = ctx.symbol("m");
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");
    let qdd = ctx.symbol("qdd");

    let half = ctx.rational(1, 2);
    let ke = &half * &m * &qd.powi(2);
    let pe = ctx.int(0);

    let eqs = euler_lagrange(&ke, &pe, &[(&q, &qd)], &[&qdd]).unwrap();
    assert_eq!(eqs.len(), 1);

    // Substitute m=3, qdd=4: expect τ = 3*4 = 12
    let val = eqs[0]
        .subs(&m, &ctx.int(3))
        .subs(&qd, &ctx.int(0))
        .subs(&qdd, &ctx.int(4))
        .subs(&q, &ctx.int(0))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(val, 12.0, 1e-12, "Free particle: m·qdd = 3*4 = 12");
}

#[test]
fn euler_lagrange_spring() {
    let ctx = Context::new();
    // Harmonic oscillator: T = ½m·q̇², V = ½k·q²
    // Equation of motion: m·q̈ + k·q = τ
    let m = ctx.symbol("m");
    let k = ctx.symbol("k");
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");
    let qdd = ctx.symbol("qdd");

    let half = ctx.rational(1, 2);
    let ke = &half * &m * &qd.powi(2);
    let pe = &half * &k * &q.powi(2);

    let eqs = euler_lagrange(&ke, &pe, &[(&q, &qd)], &[&qdd]).unwrap();
    assert_eq!(eqs.len(), 1);

    // At m=2, k=5, q=3, qdd=4, qd=0: expect 2*4 + 5*3 = 8+15 = 23
    let val = eqs[0]
        .subs(&m, &ctx.int(2))
        .subs(&k, &ctx.int(5))
        .subs(&q, &ctx.int(3))
        .subs(&qd, &ctx.int(0))
        .subs(&qdd, &ctx.int(4))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(val, 23.0, 1e-12, "Spring: m·qdd + k·q = 2*4+5*3=23");
}

#[test]
fn euler_lagrange_pendulum() {
    let ctx = Context::new();
    // Simple pendulum: T = ½mL²q̇², V = -mgLcos(q)
    // EOM: mL²q̈ + mgLsin(q) = τ
    let m_sym = ctx.symbol("m");
    let g_sym = ctx.symbol("g");
    let l_sym = ctx.symbol("L");
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");
    let qdd = ctx.symbol("qdd");

    let half = ctx.rational(1, 2);
    let ke = &half * &m_sym * &l_sym.powi(2) * &qd.powi(2);
    let pe = -(&m_sym * &g_sym * &l_sym * &q.cos());

    let eqs = euler_lagrange(&ke, &pe, &[(&q, &qd)], &[&qdd]).unwrap();
    assert_eq!(eqs.len(), 1);

    // Numeric check: q=0.5, qd=0.1, qdd=0, m=1, L=1, g=9.8
    // Expected: mL²·qdd + mgL·sin(q) = 0 + 1*9.8*1*sin(0.5) = 9.8*0.4794... ≈ 4.6983...
    let q_val = ctx.rational(1, 2);
    let result = eqs[0]
        .subs(&m_sym, &ctx.int(1))
        .subs(&l_sym, &ctx.int(1))
        .subs(&g_sym, &ctx.rational(49, 5)) // 9.8
        .subs(&q, &q_val)
        .subs(&qd, &ctx.rational(1, 10))
        .subs(&qdd, &ctx.int(0))
        .eval()
        .eval_f64()
        .unwrap();

    // Expected: 9.8 * sin(0.5)
    let expected = 9.8 * (0.5_f64).sin();
    assert_near(result, expected, 1e-6, "Pendulum EOM at given values");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. mass_matrix tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mass_matrix_single_dof() {
    let ctx = Context::new();
    // T = ½m·q̇²  →  M = [[m]]
    let m = ctx.symbol("m");
    let qd = ctx.symbol("qd");

    let half = ctx.rational(1, 2);
    let ke = &half * &m * &qd.powi(2);
    let mm = mass_matrix(&ke, &[&qd]).unwrap();

    assert_eq!(mm.shape(), (1, 1));

    // M[0,0] should be m; substitute m=7, expect 7
    let val = mm
        .get(0, 0)
        .subs(&m, &ctx.int(7))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(val, 7.0, 1e-12, "M[0,0] = m = 7");
}

#[test]
fn mass_matrix_two_dof() {
    let ctx = Context::new();
    // Simple 2-DOF: T = ½m₁·q̇₁² + ½m₂·(q̇₁² + q̇₂² + 2·q̇₁·q̇₂·cos(q₂))
    // This is a simplified 2-link arm kinetic energy.
    // M = [[m₁ + m₂,  m₂·cos(q₂)],
    //      [m₂·cos(q₂),     m₂    ]]
    let m1 = ctx.symbol("m1");
    let m2 = ctx.symbol("m2");
    let _q1 = ctx.symbol("q1");
    let q2 = ctx.symbol("q2");
    let q1d = ctx.symbol("q1d");
    let q2d = ctx.symbol("q2d");

    let half = ctx.rational(1, 2);
    let ke = &half * &m1 * &q1d.powi(2)
        + &half * &m2 * &(&q1d.powi(2) + &q2d.powi(2) + &(2 * &q1d * &q2d * &q2.cos()));

    let mm = mass_matrix(&ke, &[&q1d, &q2d]).unwrap();
    assert_eq!(mm.shape(), (2, 2));

    // M[0,0] = m1 + m2
    let m00 = mm
        .get(0, 0)
        .subs(&m1, &ctx.int(3))
        .subs(&m2, &ctx.int(2))
        .subs(&q2, &ctx.int(0))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(m00, 5.0, 1e-12, "M[0,0] = m1+m2 = 5");

    // M[1,1] = m2
    let m11 = mm
        .get(1, 1)
        .subs(&m2, &ctx.int(2))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(m11, 2.0, 1e-12, "M[1,1] = m2 = 2");

    // M[0,1] = m2·cos(q2); at q2=0 → m2=2
    let m01 = mm
        .get(0, 1)
        .subs(&m2, &ctx.int(2))
        .subs(&q2, &ctx.int(0))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(m01, 2.0, 1e-12, "M[0,1] = m2·cos(0) = 2");

    // Verify symmetry: M[1,0] = M[0,1]
    let m10 = mm
        .get(1, 0)
        .subs(&m2, &ctx.int(2))
        .subs(&q2, &ctx.int(0))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(m10, m01, 1e-12, "M[1,0] = M[0,1] (symmetry)");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. gravity_vector tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gravity_vector_pendulum() {
    let ctx = Context::new();
    // V = m·g·L·cos(q)  →  g(q) = ∂V/∂q = -m·g·L·sin(q)
    let m = ctx.symbol("m");
    let g = ctx.symbol("g");
    let l = ctx.symbol("L");
    let q = ctx.symbol("q");

    let pe = &m * &g * &l * &q.cos();
    let gv = gravity_vector(&pe, &[&q]);
    assert_eq!(gv.len(), 1);

    // At q=π/6, m=1, g=10, L=2: dV/dq = -1*10*2*sin(π/6) = -20*0.5 = -10
    let pi_over_6 = &ctx.pi() / 6;
    let val = gv[0]
        .subs(&m, &ctx.int(1))
        .subs(&g, &ctx.int(10))
        .subs(&l, &ctx.int(2))
        .subs(&q, &pi_over_6)
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(val, -10.0, 1e-6, "dV/dq = -m*g*L*sin(pi/6) = -10");
}

#[test]
fn gravity_vector_zero_potential() {
    let ctx = Context::new();
    // V = 0 → g(q) = [0, 0]
    let q1 = ctx.symbol("q1");
    let q2 = ctx.symbol("q2");

    let pe = ctx.int(0);
    let gv = gravity_vector(&pe, &[&q1, &q2]);
    assert_eq!(gv.len(), 2);

    let g0 = gv[0].eval().eval_f64().unwrap();
    let g1 = gv[1].eval().eval_f64().unwrap();
    assert_near(g0, 0.0, 1e-12, "g[0] = 0 for zero potential");
    assert_near(g1, 0.0, 1e-12, "g[1] = 0 for zero potential");
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. coriolis_matrix tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn coriolis_matrix_constant_mass() {
    let ctx = Context::new();
    // If M is constant (doesn't depend on q), then C = 0.
    // T = ½m·q̇² → M = [[m]], all ∂M/∂q = 0, so C = [[0]]
    let m = ctx.symbol("m");
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");

    let half = ctx.rational(1, 2);
    let ke = &half * &m * &qd.powi(2);
    let mm = mass_matrix(&ke, &[&qd]).unwrap();
    let c = coriolis_matrix(&mm, &[&q], &[&qd]).unwrap();

    assert_eq!(c.shape(), (1, 1));

    // C[0,0] should be 0 regardless of values
    let val = c
        .get(0, 0)
        .subs(&m, &ctx.int(5))
        .subs(&q, &ctx.int(1))
        .subs(&qd, &ctx.int(2))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(val, 0.0, 1e-12, "C[0,0] = 0 for constant mass");
}

#[test]
fn coriolis_matrix_two_dof() {
    let ctx = Context::new();
    // For the 2-DOF system:
    // T = ½m₁·q̇₁² + ½m₂·(q̇₁² + q̇₂² + 2·q̇₁·q̇₂·cos(q₂))
    // M = [[m₁+m₂, m₂cos(q₂)], [m₂cos(q₂), m₂]]
    //
    // Christoffel: Γ₁₂₂ = ½(∂M₁₂/∂q₂ + ∂M₁₂/∂q₂ - ∂M₂₂/∂q₁)
    //            = ½(-m₂sin(q₂) + (-m₂sin(q₂)) - 0) = -m₂sin(q₂)
    // C₁₂ = Γ₁₂₂·q̇₂ + Γ₁₂₁·q̇₁
    //      = -m₂sin(q₂)·q̇₂ + 0   (since Γ₁₂₁ involves derivatives wrt q₁, all zero)
    //
    // So C₁₂ = -m₂·sin(q₂)·q̇₂
    let m1 = ctx.symbol("m1");
    let m2 = ctx.symbol("m2");
    let q1 = ctx.symbol("q1");
    let q2 = ctx.symbol("q2");
    let q1d = ctx.symbol("q1d");
    let q2d = ctx.symbol("q2d");

    let half = ctx.rational(1, 2);
    let ke = &half * &m1 * &q1d.powi(2)
        + &half * &m2 * &(&q1d.powi(2) + &q2d.powi(2) + &(2 * &q1d * &q2d * &q2.cos()));

    let mm = mass_matrix(&ke, &[&q1d, &q2d]).unwrap();
    let c = coriolis_matrix(&mm, &[&q1, &q2], &[&q1d, &q2d]).unwrap();

    assert_eq!(c.shape(), (2, 2));

    // Test C[0,1] = -m₂·sin(q₂)·q̇₂
    // At m2=3, q2=π/6, q2d=2: C₀₁ = -3·sin(π/6)·2 = -3·0.5·2 = -3
    let pi_over_6 = &ctx.pi() / 6;
    let c01 = c
        .get(0, 1)
        .subs(&m1, &ctx.int(1))
        .subs(&m2, &ctx.int(3))
        .subs(&q1, &ctx.int(0))
        .subs(&q2, &pi_over_6)
        .subs(&q1d, &ctx.int(0))
        .subs(&q2d, &ctx.int(2))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(c01, -3.0, 1e-6, "C[0,1] = -m2*sin(q2)*q2d = -3");
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. christoffel_symbols tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn christoffel_symbols_constant_mass() {
    let ctx = Context::new();
    // Constant mass matrix → all Christoffel symbols are zero
    let m = ctx.symbol("m");
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");

    let half = ctx.rational(1, 2);
    let ke = &half * &m * &qd.powi(2);
    let mm = mass_matrix(&ke, &[&qd]).unwrap();
    let cs = christoffel_symbols(&mm, &[&q]).unwrap();

    assert_eq!(cs.len(), 1);
    assert_eq!(cs[0].len(), 1);
    assert_eq!(cs[0][0].len(), 1);

    let val = cs[0][0][0].eval().eval_f64().unwrap();
    assert_near(val, 0.0, 1e-12, "Christoffel[0][0][0] = 0 for constant M");
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. manipulator_equation tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn manipulator_equation_free_particle() {
    let ctx = Context::new();
    // Free particle: T = ½m·q̇², V = 0
    // M = [[m]], C = [[0]], g = [0]
    let m = ctx.symbol("m");
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");

    let half = ctx.rational(1, 2);
    let ke = &half * &m * &qd.powi(2);
    let pe = ctx.int(0);

    let (mass, coriolis, grav) = manipulator_equation(&ke, &pe, &[&q], &[&qd]).unwrap();

    assert_eq!(mass.shape(), (1, 1));
    assert_eq!(coriolis.shape(), (1, 1));
    assert_eq!(grav.len(), 1);

    // M[0,0] = m → 5
    let m_val = mass
        .get(0, 0)
        .subs(&m, &ctx.int(5))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(m_val, 5.0, 1e-12, "M[0,0] = m = 5");

    // C[0,0] = 0
    let c_val = coriolis
        .get(0, 0)
        .subs(&m, &ctx.int(5))
        .subs(&q, &ctx.int(1))
        .subs(&qd, &ctx.int(1))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(c_val, 0.0, 1e-12, "C[0,0] = 0");

    // g[0] = 0
    let g_val = grav[0].eval().eval_f64().unwrap();
    assert_near(g_val, 0.0, 1e-12, "g[0] = 0");
}

#[test]
fn manipulator_equation_spring_pendulum() {
    let ctx = Context::new();
    // Spring: T = ½m·q̇², V = ½k·q²
    // M = [[m]], C = [[0]], g = [k·q]
    let m = ctx.symbol("m");
    let k = ctx.symbol("k");
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");

    let half = ctx.rational(1, 2);
    let ke = &half * &m * &qd.powi(2);
    let pe = &half * &k * &q.powi(2);

    let (mass, _coriolis, grav) = manipulator_equation(&ke, &pe, &[&q], &[&qd]).unwrap();

    // M[0,0] = m
    let m_val = mass
        .get(0, 0)
        .subs(&m, &ctx.int(4))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(m_val, 4.0, 1e-12, "M = m = 4");

    // g[0] = ∂V/∂q = k·q
    // At k=3, q=2: g = 6
    let g_val = grav[0]
        .subs(&k, &ctx.int(3))
        .subs(&q, &ctx.int(2))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(g_val, 6.0, 1e-12, "g[0] = k*q = 6");
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Euler-Lagrange matches M·q̈ + C·q̇ + g for a nontrivial system
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn euler_lagrange_matches_manipulator_equation() {
    let ctx = Context::new();
    // For a 1-DOF pendulum:
    //   T = ½mL²q̇², V = -mgLcos(q)
    //   EL gives: mL²q̈ + mgLsin(q)
    //   M·q̈ + C·q̇ + g should give the same result
    let m_sym = ctx.symbol("m");
    let g_sym = ctx.symbol("g");
    let l_sym = ctx.symbol("L");
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");
    let qdd = ctx.symbol("qdd");

    let half = ctx.rational(1, 2);
    let ke = &half * &m_sym * &l_sym.powi(2) * &qd.powi(2);
    let pe = -(&m_sym * &g_sym * &l_sym * &q.cos());

    // Euler-Lagrange
    let eqs = euler_lagrange(&ke, &pe, &[(&q, &qd)], &[&qdd]).unwrap();

    // Manipulator equation
    let (mass, coriolis, grav) = manipulator_equation(&ke, &pe, &[&q], &[&qd]).unwrap();

    // M·q̈ + C·q̇ + g for 1-DOF: M[0,0]·qdd + C[0,0]·qd + g[0]
    let manip_result = &(mass.get(0, 0) * &qdd) + &(&(coriolis.get(0, 0) * &qd) + &grav[0]);

    // Evaluate both at specific values
    let m_val = ctx.int(2);
    let g_val = ctx.int(10);
    let l_val = ctx.int(1);
    let q_val = ctx.rational(1, 3);
    let qd_val = ctx.rational(1, 5);
    let qdd_val = ctx.int(3);

    let el_num = eqs[0]
        .subs(&m_sym, &m_val)
        .subs(&g_sym, &g_val)
        .subs(&l_sym, &l_val)
        .subs(&q, &q_val)
        .subs(&qd, &qd_val)
        .subs(&qdd, &qdd_val)
        .eval()
        .eval_f64()
        .unwrap();

    let manip_num = manip_result
        .subs(&m_sym, &m_val)
        .subs(&g_sym, &g_val)
        .subs(&l_sym, &l_val)
        .subs(&q, &q_val)
        .subs(&qd, &qd_val)
        .subs(&qdd, &qdd_val)
        .eval()
        .eval_f64()
        .unwrap();

    assert_near(
        el_num,
        manip_num,
        1e-6,
        "Euler-Lagrange and M·qdd+C·qd+g must agree",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn total_time_derivative_of_velocity() {
    let ctx = Context::new();
    // d/dt(q̇) = q̈
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");
    let qdd = ctx.symbol("qdd");

    let result = total_time_derivative(&qd, &[(&q, &qd)], &[&qdd]).unwrap();
    // Should be qdd; substitute qdd=42, expect 42
    let val = result
        .subs(&qdd, &ctx.int(42))
        .subs(&qd, &ctx.int(0))
        .subs(&q, &ctx.int(0))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(val, 42.0, 1e-12, "d/dt(qd) = qdd = 42");
}

#[test]
fn total_time_derivative_mixed() {
    let ctx = Context::new();
    // d/dt(q · q̇) = q̇ · q̇ + q · q̈ = q̇² + q·q̈
    let q = ctx.symbol("q");
    let qd = ctx.symbol("qd");
    let qdd = ctx.symbol("qdd");

    let expr = &q * &qd;
    let result = total_time_derivative(&expr, &[(&q, &qd)], &[&qdd]).unwrap();

    // At q=2, qd=3, qdd=5: expect 3² + 2·5 = 9 + 10 = 19
    let val = result
        .subs(&q, &ctx.int(2))
        .subs(&qd, &ctx.int(3))
        .subs(&qdd, &ctx.int(5))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(val, 19.0, 1e-12, "d/dt(q*qd) = qd²+q*qdd = 19");
}

#[test]
fn mass_matrix_is_symmetric() {
    let ctx = Context::new();
    // For any well-formed kinetic energy, M should be symmetric.
    let m1 = ctx.symbol("m1");
    let m2 = ctx.symbol("m2");
    let q2 = ctx.symbol("q2");
    let q1d = ctx.symbol("q1d");
    let q2d = ctx.symbol("q2d");

    let half = ctx.rational(1, 2);
    let ke = &half * &m1 * &q1d.powi(2)
        + &half * &m2 * &(&q1d.powi(2) + &q2d.powi(2) + &(2 * &q1d * &q2d * &q2.cos()));

    let mm = mass_matrix(&ke, &[&q1d, &q2d]).unwrap();

    // Evaluate M[0,1] and M[1,0] at specific point
    let pi_over_4 = &ctx.pi() / 4;
    let m01 = mm
        .get(0, 1)
        .subs(&m1, &ctx.int(1))
        .subs(&m2, &ctx.int(3))
        .subs(&q2, &pi_over_4)
        .eval()
        .eval_f64()
        .unwrap();
    let m10 = mm
        .get(1, 0)
        .subs(&m1, &ctx.int(1))
        .subs(&m2, &ctx.int(3))
        .subs(&q2, &pi_over_4)
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(
        m01,
        m10,
        1e-12,
        "Mass matrix must be symmetric: M[0,1] = M[1,0]",
    );
}

#[test]
fn gravity_vector_two_dof() {
    let ctx = Context::new();
    // V = m1·g·L1·cos(q1) + m2·g·(L1·cos(q1) + L2·cos(q1+q2))
    // g[0] = ∂V/∂q1 = -(m1+m2)·g·L1·sin(q1) - m2·g·L2·sin(q1+q2)
    // g[1] = ∂V/∂q2 = -m2·g·L2·sin(q1+q2)
    let m1 = ctx.symbol("m1");
    let m2 = ctx.symbol("m2");
    let g_sym = ctx.symbol("g");
    let l1 = ctx.symbol("L1");
    let l2 = ctx.symbol("L2");
    let q1 = ctx.symbol("q1");
    let q2 = ctx.symbol("q2");

    let pe = &m1 * &g_sym * &l1 * &q1.cos()
        + &m2 * &g_sym * &(&l1 * &q1.cos() + &l2 * &(&q1 + &q2).cos());

    let gv = gravity_vector(&pe, &[&q1, &q2]);
    assert_eq!(gv.len(), 2);

    // At q1=0, q2=0, all sines are 0 so gravity is also 0 at this config
    // Actually sin(0)=0, so ∂V/∂q evaluated at 0 gives 0
    let g0 = gv[0]
        .subs(&m1, &ctx.int(1))
        .subs(&m2, &ctx.int(1))
        .subs(&g_sym, &ctx.int(10))
        .subs(&l1, &ctx.int(1))
        .subs(&l2, &ctx.int(1))
        .subs(&q1, &ctx.int(0))
        .subs(&q2, &ctx.int(0))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(g0, 0.0, 1e-10, "g[0] at q1=q2=0 should be 0 (sin(0)=0)");

    // At q1=π/2, q2=0: sin(q1)=1, sin(q1+q2)=1
    // g[0] = -(1+1)·10·1·1 - 1·10·1·1 = -20-10 = -30
    let pi_over_2 = &ctx.pi() / 2;
    let g0_pi2 = gv[0]
        .subs(&m1, &ctx.int(1))
        .subs(&m2, &ctx.int(1))
        .subs(&g_sym, &ctx.int(10))
        .subs(&l1, &ctx.int(1))
        .subs(&l2, &ctx.int(1))
        .subs(&q1, &pi_over_2)
        .subs(&q2, &ctx.int(0))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(g0_pi2, -30.0, 1e-6, "g[0] at q1=π/2,q2=0 = -30");

    // g[1] at same config: -m2·g·L2·sin(π/2+0) = -10
    let g1_pi2 = gv[1]
        .subs(&m1, &ctx.int(1))
        .subs(&m2, &ctx.int(1))
        .subs(&g_sym, &ctx.int(10))
        .subs(&l1, &ctx.int(1))
        .subs(&l2, &ctx.int(1))
        .subs(&q1, &pi_over_2)
        .subs(&q2, &ctx.int(0))
        .eval()
        .eval_f64()
        .unwrap();
    assert_near(g1_pi2, -10.0, 1e-6, "g[1] at q1=π/2,q2=0 = -10");
}
