//! The applied-domains hunt: control systems (`symplex::control`),
//! quaternions, vector calculus and Lagrangian dynamics.

use symplex::control::TransferFunction;
use symplex::dynamics::{
    GeneralizedCoordinate, euler_lagrange, manipulator_equation, mass_matrix, total_time_derivative,
};
use symplex::prelude::*;
use symplex::quaternion::EulerConvention;
use symplex::robotics::{rot_euler, rot_z};
use symplex::vector::{
    is_conservative, is_solenoidal, line_integral_scalar, line_integral_vector, scalar_potential,
};

fn close(actual: &Ex, want: f64, rel: f64, label: &str) {
    let v = actual
        .eval_f64()
        .unwrap_or_else(|e| panic!("{label}: `{actual}` does not evaluate: {e}"));
    assert!(
        (v - want).abs() <= rel * (1.0 + want.abs()),
        "{label}: got {v}, want {want}"
    );
}

/// Poles and zeros are listed with multiplicity, as SymPy 1.14's
/// `TransferFunction(1, (s + 1)**2, s).poles()` = `[-1, -1]` and
/// `TransferFunction((s - 2)**3, (s + 1)*(s + 3), s).zeros()` = `[2, 2, 2]`
/// do, and as `StateSpace::poles` of the realisation does.  Before, the
/// solver's distinct roots came back: `[-1]` and `[2]` (the hunter: 114
/// of 300 random transfer functions had too few poles).
#[test]
fn transfer_function_poles_and_zeros_keep_multiplicity() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    let g = TransferFunction::new(ctx.int(1), (&s + 1).powi(2), s.clone());
    assert_eq!(g.poles(), vec![ctx.int(-1), ctx.int(-1)]);
    let h = TransferFunction::new((&s - 2).powi(3), (&s + 1) * (&s + 3), s.clone());
    assert_eq!(h.zeros(), vec![ctx.int(2); 3]);
    let mut p: Vec<String> = h.poles().iter().map(|e| e.to_string()).collect();
    p.sort();
    assert_eq!(p, ["-1", "-3"]);
    // The poles of the realisation agree.
    let ss = g.to_state_space().unwrap();
    assert_eq!(ss.poles(), g.poles());
    // An irreducible cubic: three poles (`RootOf`), each a root.
    let k = TransferFunction::new(ctx.int(1), &(&s.powi(3) + &s) + 1, s.clone());
    let kp = k.poles();
    assert_eq!(kp.len(), 3);
    for r in &kp {
        let v = (&(&r.powi(3) + r) + 1).eval_complex64().unwrap();
        assert!(v.norm() < 1e-12, "{r}");
    }
}

/// The DC gain is `lim_{s→0} G(s)` (SymPy: `TransferFunction(s, s**2 + s,
/// s).dc_gain()` = `1`).  Before, a pole-zero cancellation at the origin
/// gave `num(0)/den(0) = 0/0 = nan` (44 of 300 random transfer functions).
#[test]
fn dc_gain_at_a_cancellation_at_the_origin() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    let tf = |n: &[i64], d: &[i64]| TransferFunction::from_coeffs(n, d, &s);
    assert_eq!(tf(&[0, 1], &[0, 1, 1]).dc_gain(), ctx.int(1));
    // s²/(s²(s + 2)) → 1/2, s²/(s(s² + 1)) → 0, s/s³ → complex infinity
    assert_eq!(tf(&[0, 0, 1], &[0, 0, 2, 1]).dc_gain(), ctx.rational(1, 2));
    assert_eq!(tf(&[0, 0, 1], &[0, 1, 0, 1]).dc_gain(), ctx.int(0));
    assert_eq!(tf(&[0, 1], &[0, 0, 0, 1]).dc_gain(), ctx.complex_infinity());
    // unchanged away from the cancellation
    assert_eq!(tf(&[5], &[2, 1]).dc_gain(), ctx.rational(5, 2));
    assert_eq!(tf(&[1], &[0, 1]).dc_gain(), ctx.complex_infinity());
    // symbolic coefficients: a·s/(a·s² + 2s) → a/2
    let a = ctx.symbol("a");
    let g = TransferFunction::new(&a * &s, &(&a * &s.powi(2)) + &(&s * 2), s.clone());
    assert_eq!(g.dc_gain(), &a / 2);
}

/// Each transfer function is a function of its own Laplace variable, so
/// `series`, `parallel` and `feedback_with` rename the other one's.  Before,
/// `G(s) = 1/s` in series with `H(p) = 1/(p + 2)` was `1/(s·(p + 2))`, `p`
/// a constant (the hunter: 37–39 of 300 per interconnection).  SymPy
/// raises a `ValueError` for mixed variables instead.
#[test]
fn interconnections_rename_the_other_laplace_variable() {
    let ctx = Context::new();
    let (s, p) = (ctx.symbol("s"), ctx.symbol("p"));
    let g = TransferFunction::new(ctx.int(1), s.clone(), s.clone());
    let h = TransferFunction::new(ctx.int(1), &p + 2, p.clone());
    let at = |tf: &TransferFunction| tf.eval_at(&ctx.int(3));
    assert_eq!(at(&g.series(&h)), ctx.rational(1, 15));
    assert_eq!(at(&g.parallel(&h)), ctx.rational(8, 15));
    // G/(1 + GH) at s = 3: (1/3)/(1 + 1/15) = 5/16
    assert_eq!(at(&g.feedback_with(&h)), ctx.rational(5, 16));
    assert!(!g.series(&h).den.free_symbols().contains(&p));
}

/// Stability of a state-space model with symbolic poles is decided by
/// Routh–Hurwitz on `det(sI − A)`, as SymPy's
/// `get_asymptotic_stability_conditions` does.  Before, every symbolic
/// pole gave `None`: `A = [[−k, 1], [0, −k − 1]]` with `k > 0` (poles `−k`,
/// `−k − 1`).  The hunter: 198 of 300 random `A₀ + k·E` undecided whose
/// samples at `k = 1, 2, 3, 5, 10, 100` all agree; now 38.  And
/// `is_routh_stable` refutes a polynomial with a coefficient of the wrong
/// sign even when the table's entries are undecidable: `s³ + (k − 1)s² −
/// k·s + 1` was `None`.
#[test]
fn stability_with_a_symbolic_parameter() {
    let ctx = Context::new();
    let k = ctx.symbol_with("k", &[Assumption::Positive]).unwrap();
    let a = Matrix::new(vec![vec![-&k, ctx.int(1)], vec![ctx.int(0), -&k - 1]]).unwrap();
    let ss = StateSpace::new(
        a,
        matrix![ctx, [0], [1]],
        matrix![ctx, [1, 0]],
        matrix![ctx, [0]],
    )
    .unwrap();
    assert_eq!(ss.is_stable(), Some(true));
    let unstable = Matrix::new(vec![vec![&k + 1, ctx.int(1)], vec![ctx.int(0), -&k]]).unwrap();
    let ss = StateSpace::new(
        unstable,
        matrix![ctx, [0], [1]],
        matrix![ctx, [1, 0]],
        matrix![ctx, [0]],
    )
    .unwrap();
    assert_eq!(ss.is_stable(), Some(false));
    let coeffs = [ctx.int(1), &k - 1, -&k, ctx.int(1)];
    assert_eq!(symplex::control::is_routh_stable(&coeffs), Some(false));
    // genuinely parameter-dependent: s³ + 3s² + 3s + 1 + k is stable for k < 8
    let coeffs = [ctx.int(1), ctx.int(3), ctx.int(3), &k + 1];
    assert_eq!(symplex::control::is_routh_stable(&coeffs), None);
}

/// Zero-order hold with a series of order above 20.  Before, `k!` was an
/// `i64` and order 21 panicked with "attempt to multiply with overflow"
/// (104 of 300 random systems in the hunter).  Reference: the exact
/// `A_d = e^{AT}`, `B_d = ∫₀ᵀ e^{Aτ}dτ·B` of `A = [[0, 1], [−2, −3]]`,
/// `T = 1/10` (scipy.signal.cont2discrete(…, 0.1, method='zoh') gives
/// `[[0.99094408, 0.08610666], [−0.17221333, 0.73262409]]`,
/// `[[0.00452796], [0.08610666]]`).
#[test]
fn zoh_series_of_high_order() {
    let ctx = Context::new();
    let ss = StateSpace::new(
        matrix![ctx, [0, 1], [-2, -3]],
        matrix![ctx, [0], [1]],
        matrix![ctx, [1, 0]],
        matrix![ctx, [0]],
    )
    .unwrap();
    let z = ss.discretize_zoh(&ctx.rational(1, 10), 25).unwrap();
    let (e1, e2) = ((-0.1f64).exp(), (-0.2f64).exp());
    let ad = [[2.0 * e1 - e2, e1 - e2], [2.0 * (e2 - e1), 2.0 * e2 - e1]];
    let bd = [(1.0 - e1) - (1.0 - e2) / 2.0, (1.0 - e2) - (1.0 - e1)];
    for (i, (row, b_i)) in ad.iter().zip(bd).enumerate() {
        for (j, a_ij) in row.iter().enumerate() {
            close(z.a().get(i, j), *a_ij, 1e-15, "A_d");
        }
        close(z.b().get(i, 0), b_i, 1e-15, "B_d");
    }
}

/// An orthogonal matrix with determinant −1 is no rotation and has no
/// quaternion.  Before, `from_rotation_matrix(diag(1, 1, −1))` returned the
/// non-unit `(√2/2, 0, 0, 0)` (every one of 300 reflected matrices in the
/// hunter was accepted).  SymPy's `Quaternion.from_rotation_matrix` does not
/// check either; scipy's `Rotation.from_matrix` takes the nearest rotation.
#[test]
fn reflection_has_no_quaternion() {
    let ctx = Context::new();
    let refl = matrix![ctx, [1, 0, 0], [0, 1, 0], [0, 0, -1]];
    assert!(Quaternion::from_rotation_matrix(&refl).is_err());
    let q = Quaternion::new(ctx.int(1), ctx.int(2), ctx.int(-2), ctx.int(4)).normalize();
    let r = q.to_rotation_matrix();
    assert!(Quaternion::from_rotation_matrix(&r.scale(&ctx.int(-1))).is_err());
    assert_eq!(
        Quaternion::from_rotation_matrix(&r).unwrap().equals(&q),
        Some(true)
    );
    // symbolic: diag(1, 1, −1)·Rz(θ) has det −1 for every θ
    let th = ctx.symbol("theta");
    let sym_refl = refl.matmul(&rot_z(&th)).unwrap();
    assert!(Quaternion::from_rotation_matrix(&sym_refl).is_err());
    assert!(Quaternion::from_rotation_matrix(&rot_z(&th)).is_ok());
}

/// A quaternion with a zero vector part is a real number and gets the
/// real power.  Before, `pow` went through `exp(t·ln q)`, and `0²` was
/// `exp(2·ln 0) = exp(zoo) = nan` (SymPy: `Quaternion(0, 0, 0, 0)**2` = 0).
#[test]
fn power_of_a_real_quaternion() {
    let ctx = Context::new();
    let zero = Quaternion::zero(&ctx);
    assert_eq!(zero.pow(&ctx.int(2)), zero);
    let two = Quaternion::new(ctx.int(2), ctx.int(0), ctx.int(0), ctx.int(0));
    assert_eq!(two.pow(&ctx.rational(1, 2)).w, ctx.int(2).sqrt());
    assert_eq!(two.pow(&ctx.int(-1)).w, ctx.rational(1, 2));
    // unchanged for a nonzero vector part: (1 + i)³ = −2 + 2i
    let q = Quaternion::new(ctx.int(1), ctx.int(1), ctx.int(0), ctx.int(0));
    let c = q.pow(&ctx.int(3));
    close(&c.w, -2.0, 1e-14, "w");
    close(&c.x, 2.0, 1e-14, "x");
}

/// Euler angles at gimbal lock with exact angles.  Before, the XYZ angles
/// of `(π/4, −π/2, −3π/4)` contained `φ = atan2(R₂₁, R₁₁)` with `R₂₁` an
/// unrecognised zero and `R₁₁ = −1`, on the branch cut of `atan2`, and
/// `φ.eval_f64()` failed with `PrecisionExhausted`.  Oracle: the angles
/// reproduce the rotation (scipy `Rotation.from_euler('XYZ', …)`).
#[test]
fn euler_angles_at_gimbal_lock_evaluate() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let (a, b, c) = (&pi / 4, -(&pi / 2), -(&(&pi * 3) / 4));
    let q = Quaternion::from_euler(&a, &b, &c, EulerConvention::XYZ);
    let (p, t, s) = q.to_euler(EulerConvention::XYZ);
    close(&p, std::f64::consts::PI, 1e-15, "phi");
    close(&t, -std::f64::consts::FRAC_PI_2, 1e-15, "theta");
    let want = rot_euler(&a, &b, &c, EulerConvention::XYZ)
        .eval_f64()
        .unwrap();
    let got = rot_euler(&p, &t, &s, EulerConvention::XYZ)
        .eval_f64()
        .unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!((got[i][j] - want[i][j]).abs() < 1e-14, "R[{i}][{j}]");
        }
    }
}

/// `slerp(q, q, t)` is the constant path `q`; before it was `0/0` (`nan` in
/// every component).
#[test]
fn slerp_of_equal_quaternions_is_constant() {
    let ctx = Context::new();
    let q = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(0), &ctx.int(1), &ctx.int(1));
    let t = ctx.symbol("t");
    assert_eq!(q.slerp(&q, &t), q);
}

/// A field whose curl is not identically zero is not conservative.  Before,
/// `is_conservative` asked whether each curl component vanishes for *every*
/// value of `x, y, z` and answered `None` for components such as
/// `x·sin x − cos x − 4y/(y² + 2)²`, which vanish somewhere (193 of 200
/// random rotational fields); a point where the component is a certified
/// nonzero constant decides it.  A component depending on a parameter
/// stays undecided.
#[test]
fn rotational_fields_are_refuted() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let v = [&x, &y, &z];
    let two_over = ctx.int(2) / &(&y.powi(2) + 2);
    let f = Matrix::col_vector(vec![
        -&x.powi(2) - &two_over,
        &(-&x * &x.cos()) + &two_over,
        z.powi(2).sin(),
    ])
    .unwrap();
    assert_eq!(is_conservative(&f, &v), Some(false));
    assert_eq!(is_solenoidal(&f, &v), Some(false));
    let a = ctx.symbol("a");
    let g = Matrix::col_vector(vec![ctx.int(0), ctx.int(0), &a * &x]).unwrap();
    assert_eq!(is_conservative(&g, &v), None);
    let vortex = Matrix::col_vector(vec![
        -&y / &(&x.powi(2) + &y.powi(2)),
        &x / &(&x.powi(2) + &y.powi(2)),
        ctx.int(0),
    ])
    .unwrap();
    assert_eq!(is_conservative(&vortex, &v), Some(true));
}

/// The scalar potential of `∇(sin(xy) + cos(yz))`.  `∫ y·cos(xy) dx` is the
/// `Piecewise` (`sin(xy)/y` for `y ≠ 0`, `x` otherwise), and the next
/// integration could not handle it: before, the call failed with "could
/// not integrate component 1 (… Piecewise …)" (16 of 200 random gradients).
/// The generic branch is taken; the result is checked by
/// re-differentiation (SymPy's `scalar_potential` carries the
/// `Piecewise`).
#[test]
fn scalar_potential_with_a_parametric_antiderivative() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let v = [&x, &y, &z];
    let phi = &(&x * &y).sin() + &(&y * &z).cos();
    let f = symplex::vector::gradient(&phi, &v).unwrap();
    let p = scalar_potential(&f, &v).unwrap();
    assert!((&p - &phi).simplify().is_zero_structural(), "{p}");
}

/// The curve is substituted simultaneously, so a curve parametrised by one
/// of the coordinates is not substituted twice.  Before, `∫ (x² + y² + 1)
/// ds` along `(x, y) = (−2y, 1)`, `y ∈ [0, 2π]` (parameter named `y`)
/// replaced `x` by `−2y` and then every `y` by `1`: `24π` instead of
/// `64π³/3 + 8π = 686.59997707511…` (mpmath quadrature agrees).
#[test]
fn line_integral_parametrised_by_a_coordinate() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f = &(&x.powi(2) + &y.powi(2)) + 1;
    let curve = [&y * -2, ctx.int(1)];
    let two_pi = &ctx.pi() * 2;
    let v = line_integral_scalar(&f, &[&x, &y], &curve, &y, &ctx.int(0), &two_pi).unwrap();
    let pi = ctx.pi();
    assert_eq!(
        v.simplify(),
        (&(&pi.powi(3) * 64) / 3 + &(&pi * 8)).simplify()
    );
    // the work of (x, x) along (y, 2y), y ∈ [0, 1]: ∫ y·1 + y·2 dy = 3/2
    let fld = Matrix::col_vector(vec![x.clone(), x.clone()]).unwrap();
    let w = line_integral_vector(
        &fld,
        &[&x, &y],
        &[y.clone(), &y * 2],
        &y,
        &ctx.int(0),
        &ctx.int(1),
    )
    .unwrap();
    assert_eq!(w, ctx.rational(3, 2));
}

/// Line integrals go through the definite integrator.  Before, they were
/// `F(b) − F(a)` of the indefinite integral, which left the unevaluable
/// `−Subs(Integral(…), t, a) + Subs(Integral(…), t, b)` when no closed form
/// exists (164 of 200 scalar and 94 of 200 vector line integrals in the
/// hunter), even for `∫₀^π |cos t| dt = 2`.  Reference for the integral
/// without closed form: mpmath `quad(lambda t: t**3*sqrt(4*sin(2*t)**2 +
/// 1), [-3, -2])` = `−29.69657710689647`.
#[test]
fn line_integrals_without_an_antiderivative() {
    let ctx = Context::new();
    let (x, y, t) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("t"));
    let len = line_integral_scalar(
        &ctx.int(1),
        &[&x, &y],
        &[t.sin(), ctx.int(0)],
        &t,
        &ctx.int(0),
        &ctx.pi(),
    )
    .unwrap();
    assert_eq!(len, ctx.int(2));
    let v = line_integral_scalar(
        &y.powi(3),
        &[&x, &y],
        &[(&t * 2).cos(), t.clone()],
        &t,
        &ctx.int(-3),
        &ctx.int(-2),
    )
    .unwrap();
    close(&v, -29.696_577_106_896_47, 1e-13, "∫ y³ ds");
    // A symbolic bound keeps the closed form of the antiderivative.
    let big_t = ctx.symbol("T");
    let s = line_integral_scalar(
        &ctx.int(1),
        &[&x, &y],
        &[&t * 3, &t * 4],
        &t,
        &ctx.int(0),
        &big_t,
    )
    .unwrap();
    assert_eq!(s, &big_t * 5);
}

/// Lagrangian dynamics with velocities given as formal derivatives,
/// `q̇ = Derivative(q, t)` (`q.formal_diff(t)`), the way the ODE solver
/// takes them.  `Ex::diff` with respect to a `Derivative` node is 0, so
/// before the pendulum `T = ½ml²q̇²`, `V = mgl(1 − cos q)` gave the
/// equation `glm·sin q − ml²·q̇·Derivative(q̇, q)` (no `ml²q̈`) and the mass
/// matrix `[[0]]`.  SymPy (`LagrangesMethod` with `dynamicsymbols`):
/// `g·l·m·sin(q) + l²·m·q''`.
#[test]
fn lagrangian_with_formal_derivative_velocities() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let q = ctx.symbol("q");
    let qd = q.formal_diff(&t);
    let qdd = qd.formal_diff(&t);
    let (m, l, g) = (ctx.symbol("m"), ctx.symbol("l"), ctx.symbol("g"));
    let ke = &(&(&m * &l.powi(2)) * &qd.powi(2)) / 2;
    let pe = &(&(&m * &g) * &l) * &(ctx.int(1) - q.cos());
    let coords = [GeneralizedCoordinate {
        q: &q,
        q_dot: &qd,
        q_ddot: &qdd,
    }];
    let el = euler_lagrange(&ke, &pe, &coords);
    let want = &(&(&(&g * &l) * &m) * &q.sin()) + &(&(&m * &l.powi(2)) * &qdd);
    assert_eq!((&el[0] - &want).expand(), ctx.int(0), "{}", el[0]);
    let mm = mass_matrix(&ke, &[&qd]).unwrap();
    assert_eq!(mm.get(0, 0), &(&m * &l.powi(2)));
    let me = manipulator_equation(&ke, &pe, &[&q], &[&qd]).unwrap();
    assert_eq!(me.mass.get(0, 0), &(&m * &l.powi(2)));
    // dE/dt = q̇·(ml²q̈ + mgl·sin q)
    let de = total_time_derivative(&(&ke + &pe), &coords);
    assert_eq!((&de - &(&qd * &want)).expand(), ctx.int(0), "{de}");
}
