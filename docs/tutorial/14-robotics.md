# Chapter 14: Robotics

> **Dimensional analysis available.** Symplex supports compile-time dimensional
> analysis for robotics workflows — lengths, angles, forces, and torques can all
> be type-checked so that, e.g., swapping a link length with a joint angle is a
> compile error. This chapter shows the untyped API first (simpler, great for
> prototyping) and adds **"With Units"** callouts showing the dimension-checked
> equivalents. See [Chapter 21: Units & Dimensional Analysis](21-units.md) for
> the full reference.

Symplex provides a symbolic robotics toolkit that covers the full pipeline from mechanism description to deployable code: Denavit-Hartenberg parameters, forward kinematics, Jacobians, inverse kinematics, Lagrangian dynamics, and Rust code generation. This chapter walks through the **flagship workflow** — define an arm, derive everything symbolically, and generate optimized numerical code.

If you've used the Robotics Toolbox for MATLAB or SymPy's mechanics module, the concepts are familiar. The difference: symplex derives everything exactly and emits compiled Rust code that runs at MHz rates in your control loop.

## The Robot: A 3-DOF Planar Arm

We'll work with a three-joint planar manipulator — simple enough to follow every step, but complex enough to demonstrate the full pipeline. Each joint is revolute, with link lengths $L_1 = 0.3$ m, $L_2 = 0.25$ m, $L_3 = 0.2$ m.

```rust
use symplex::prelude::*;
use symplex::robotics::*;
use symplex::matrix::jacobian;
use symplex::vars;

vars!(theta1, theta2, theta3);

// Link lengths as exact rationals — no floating-point until the very end
let l1 = symplex::rational(3, 10);  // 0.3 m
let l2 = symplex::rational(1, 4);   // 0.25 m
let l3 = symplex::rational(1, 5);   // 0.2 m

println!("Total reach: {} m", &(&l1 + &l2) + &l3);
// Total reach: 3/4 m  (exactly 0.75 m)
```

Using `symplex::rational()` instead of `f64` keeps the entire derivation in exact arithmetic. The generated code at the end will use `f64`, but the symbolic pipeline never accumulates rounding error.

## Step 1: DH Parameters

The Denavit-Hartenberg convention describes each joint with four parameters: $(\theta, d, a, \alpha)$. For a planar arm, $d = 0$ and $\alpha = 0$ for every joint — only the joint angle $\theta$ and link length $a$ vary:

```rust
let zero = symplex::int(0);

// DH parameters: (theta, d, a, alpha)
let dh: [(&Ex, &Ex, &Ex, &Ex); 3] = [
    (&theta1, &zero, &l1, &zero),  // Joint 1
    (&theta2, &zero, &l2, &zero),  // Joint 2
    (&theta3, &zero, &l3, &zero),  // Joint 3
];

println!("DH parameters:");
for (i, (th, d, a, al)) in dh.iter().enumerate() {
    println!("  Joint {}: θ={th}, d={d}, a={a}, α={al}", i + 1);
}
```

This is the standard DH convention. Each row defines a homogeneous transformation matrix $T_i$, and the overall forward kinematics is the chain product $T_0^3 = T_1 \cdot T_2 \cdot T_3$.

### With Units

The untyped API above is convenient, but nothing prevents you from accidentally
writing `(&theta1, &zero, &theta2, &zero)` — swapping an angle for a length.
The typed API catches this at compile time:

```rust
use symplex::units::*;
use symplex::robotics::fk_position_typed;

// Lengths are Length, angles are Angle — distinct types
let l1 = Length::rational(3, 10);   // 0.3 m — dimension-checked!
let l2 = Length::rational(1, 4);    // 0.25 m
let l3 = Length::rational(1, 5);    // 0.2 m

let theta1 = Angle::symbol("theta1");
let theta2 = Angle::symbol("theta2");
let theta3 = Angle::symbol("theta3");

let zero_l = Length::zero();
let zero_a = Angle::zero();

// DH parameters: (theta: Angle, d: Length, a: Length, alpha: Angle)
let (px, py, pz) = fk_position_typed(&[
    (&theta1, &zero_l, &l1, &zero_a),  // Joint 1
    (&theta2, &zero_l, &l2, &zero_a),  // Joint 2
    (&theta3, &zero_l, &l3, &zero_a),  // Joint 3
]);
// px, py, pz are Length — guaranteed at compile time
// Swapping l1 and theta1 would be a compile error!

println!("px = {px}");  // still prints the symbolic expression
```

The typed and untyped APIs produce identical symbolic expressions — the types
are erased during code generation. Use whichever feels right for your project;
you can mix them freely (extract the inner `Ex` with `.into_inner()` if needed).

## Step 2: Forward Kinematics

The `fk_position` function chain-multiplies the DH transformation matrices and extracts the end-effector position:

```rust
let (px, py, _pz) = fk_position(&dh);

println!("End-effector position (symbolic):");
println!("  px = {px}");
println!("  py = {py}");
```

For a planar arm, the result is:

$$p_x = L_1\cos(\theta_1) + L_2\cos(\theta_1 + \theta_2) + L_3\cos(\theta_1 + \theta_2 + \theta_3)$$
$$p_y = L_1\sin(\theta_1) + L_2\sin(\theta_1 + \theta_2) + L_3\sin(\theta_1 + \theta_2 + \theta_3)$$

The exact symbolic form depends on how symplex canonicalizes the expanded trig expressions, but it's mathematically equivalent. You can also get the LaTeX for documentation:

```rust
println!("LaTeX:");
println!("  p_x = {}", px.to_latex());
println!("  p_y = {}", py.to_latex());
```

### Full Transformation Chain

If you need the full 4×4 homogeneous transformation (for orientation as well as position):

```rust
let t_full = fk_chain(&dh);
println!("T(4×4) shape: {:?}", t_full.shape());

let rot = fk_rotation(&dh);
println!("Rotation (3×3):\n{rot}");
```

For a planar arm the rotation matrix encodes the end-effector orientation about the z-axis.

## Step 3: Compute the Jacobian

The Jacobian $J$ maps joint velocities to end-effector velocities:

$$\begin{bmatrix} \dot{x} \\ \dot{y} \end{bmatrix} = J(\theta) \begin{bmatrix} \dot{\theta}_1 \\ \dot{\theta}_2 \\ \dot{\theta}_3 \end{bmatrix}$$

where $J_{ij} = \partial p_i / \partial \theta_j$. The `jacobian()` function computes each partial derivative symbolically:

```rust
let jac = jacobian(&[&px, &py], &[&theta1, &theta2, &theta3]);
println!("Jacobian (2×3):\n{jac}");

// Inspect individual entries
for i in 0..2 {
    for j in 0..3 {
        let label = if i == 0 { "x" } else { "y" };
        println!("  ∂p{label}/∂θ{} = {}", j + 1, jac.get(i, j));
    }
}
```

Each entry is a trigonometric expression. For example, $\partial p_x / \partial \theta_1$ involves $-L_1\sin(\theta_1) - L_2\sin(\theta_1+\theta_2) - L_3\sin(\theta_1+\theta_2+\theta_3)$ — the derivative of the full FK chain with respect to the first joint angle.

### Why Symbolic Jacobians Matter

A numerical Jacobian (finite differences) introduces $O(h)$ or $O(h^2)$ approximation error and requires $2n$ or $n$ FK evaluations per update. A symbolic Jacobian, once generated as code, is **exact** and evaluates in a single pass through the generated function. For a 6-DOF arm, the symbolic approach is both faster and more accurate.

## Step 4: Generate Code

This is the payoff. `to_rust_fn()` takes the symbolic Jacobian and emits an optimized Rust function with:

- **Cross-entry CSE** — shared trig calls like `sin(theta1 + theta2)` are computed once across all 6 matrix entries
- **Constant folding** — `sin(0) → 0`, `cos(0) → 1` at codegen time
- **Clean decimals** — `0.3`, not `0.30000000000000004`

```rust
let code = jac
    .to_rust_fn("robot_jacobian", &["theta1", "theta2", "theta3"])
    .expect("codegen failed");

println!("{code}");
```

The output is a complete, compilable Rust function:

```rust
/// Auto-generated by symplex — do not edit.
pub fn robot_jacobian(theta1: f64, theta2: f64, theta3: f64) -> [f64; 6] {
    let _s0 = theta1 + theta2;
    let _s1 = _s0.sin();
    let _s2 = _s0 + theta3;
    let _s3 = _s2.sin();
    let _s4 = _s2.cos();
    let _s5 = _s0.cos();
    // ... (CSE variables)
    [
        /* J[0,0] */ -0.3 * theta1.sin() - 0.25 * _s1 - 0.2 * _s3,
        /* J[0,1] */ -0.25 * _s1 - 0.2 * _s3,
        /* J[0,2] */ -0.2 * _s3,
        /* J[1,0] */  0.3 * theta1.cos() + 0.25 * _s5 + 0.2 * _s4,
        /* J[1,1] */  0.25 * _s5 + 0.2 * _s4,
        /* J[1,2] */  0.2 * _s4,
    ]
}
```

Notice how `sin(theta1 + theta2)` appears once as `_s1`, reused across four entries. This is CSE doing real work.

### Code for FK Position

You can also generate standalone position functions:

```rust
let px_code = px
    .to_rust_fn("fk_x", &["theta1", "theta2", "theta3"])
    .expect("codegen failed");
println!("{px_code}");

let py_code = py
    .to_rust_fn("fk_y", &["theta1", "theta2", "theta3"])
    .expect("codegen failed");
println!("{py_code}");
```

### Embedded Target: `f32` and `no_std`

For microcontroller deployment, generate `f32` code with the embedded preset:

```rust
let embedded_code = jac
    .to_rust_fn_with_options(
        "robot_jacobian_f32",
        &["theta1", "theta2", "theta3"],
        &symplex::matrix::CodegenOptions::embedded_f32(),
    )
    .expect("codegen failed");
println!("{embedded_code}");
```

This emits `f32` arithmetic with `libm` calls instead of `std` — ready for `#![no_std]` firmware.

## Step 5: Verify Numerically

**Always verify generated code against ground truth.** Substitute numerical joint angles into both the symbolic expression and a direct f64 computation, then compare:

```rust
let test_configs: &[(f64, f64, f64)] = &[
    (0.0, 0.0, 0.0),                          // fully extended along +x
    (std::f64::consts::FRAC_PI_4, 0.0, 0.0),  // 45° first joint
    (0.5, 0.3, 0.1),                           // arbitrary
];

for (t1, t2, t3) in test_configs {
    // Ground truth: direct f64 trig
    let gt_x = 0.3 * t1.cos() + 0.25 * (t1 + t2).cos() + 0.2 * (t1 + t2 + t3).cos();
    let gt_y = 0.3 * t1.sin() + 0.25 * (t1 + t2).sin() + 0.2 * (t1 + t2 + t3).sin();

    // Symbolic evaluation via compile()
    if let Some(px_fn) = px.compile(&["theta1", "theta2", "theta3"]) {
        let sym_x = px_fn(&[*t1, *t2, *t3]);
        let err = (sym_x - gt_x).abs();
        println!("Config ({t1:.2}, {t2:.2}, {t3:.2}): Δx = {err:.2e}");
        assert!(err < 1e-10, "FK mismatch!");
    }
}
```

If the error is zero (or within machine epsilon), your symbolic derivation and codegen are correct. This verification step catches sign errors, convention mismatches, and codegen bugs before they reach your robot.

## Inverse Kinematics (2-DOF)

For a 2-DOF planar arm, symplex can solve the inverse kinematics problem exactly using Gröbner bases over the sin/cos polynomial ring:

```rust
use symplex::robotics::inverse_kinematics_2dof;

let l1: f64 = 1.0;
let l2: f64 = 0.8;

// Axis-aligned target: algebraic solver finds exact solutions
let solutions = inverse_kinematics_2dof(l1, l2, 1.8, 0.0);
for (i, (t1, t2)) in solutions.iter().enumerate() {
    println!("Solution {}: θ₁ = {t1:.4}, θ₂ = {t2:.4}", i + 1);

    // Verify with forward kinematics
    let fx = l1 * t1.cos() + l2 * (t1 + t2).cos();
    let fy = l1 * t1.sin() + l2 * (t1 + t2).sin();
    let err = ((fx - 1.8).powi(2) + fy.powi(2)).sqrt();
    println!("  FK verify: ({fx:.6}, {fy:.6}), error = {err:.2e}");
}
```

The Gröbner-basis solver works in exact arithmetic, so it finds all solution branches (elbow-up and elbow-down) without iterative convergence issues. For targets with irrational joint angles, the algebraic solver may return empty — fall back to the standard geometric method in that case.

## Lagrangian Dynamics

For model-based control, you need the equations of motion. Symplex derives them from the kinetic and potential energy using the Euler-Lagrange equations.

### Simple Pendulum (1-DOF)

```rust
use symplex::dynamics::*;
use symplex::vars;

vars!(q, qd, qdd);
let m = symplex::var("m");
let l = symplex::var("L");
let g = symplex::var("g");

let half = symplex::half();

// T = ½·m·L²·q̇²
let ke = &half * &m * &l.powi(2) * &qd.powi(2);
// V = -m·g·L·cos(q)
let pe = &(-&m) * &g * &l * &q.cos();

println!("T = {ke}");
println!("V = {pe}");

// Euler-Lagrange: d/dt(∂L/∂q̇) - ∂L/∂q = τ
let eqs = euler_lagrange(&ke, &pe, &[(&q, &qd)], &[&qdd]);
println!("τ = {}", eqs[0]);
// τ = m·L²·q̈ + m·g·L·sin(q)
```

The result is exactly what you'd derive by hand: inertial torque plus gravitational torque.

### Double Pendulum (2-DOF) — Full Manipulator Equation

The double pendulum is the canonical 2-DOF system. The full manipulator equation is:

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau$$

```rust
vars!(q1, q2, qd1, qd2, qdd1, qdd2);
let m1 = symplex::var("m1");
let m2 = symplex::var("m2");
let l1 = symplex::var("L1");
let l2 = symplex::var("L2");
let g = symplex::var("g");
let half = symplex::half();

// Kinetic energy (standard double-pendulum form)
let ke_1 = &half * &m1 * &l1.powi(2) * &qd1.powi(2);
let ke_2_a = &half * &m2 * &l1.powi(2) * &qd1.powi(2);
let ke_2_b = &half * &m2 * &l2.powi(2) * &qd2.powi(2);
let ke_2_c = &m2 * &l1 * &l2 * &qd1 * &qd2 * &(&q1 - &q2).cos();
let ke = &(&(&ke_1 + &ke_2_a) + &ke_2_b) + &ke_2_c;

// Potential energy
let pe = &(&(-&(&m1 + &m2)) * &g * &l1 * &q1.cos())
       + &(&(-&m2) * &g * &l2 * &q2.cos());

// Euler-Lagrange equations for the double pendulum
let eqs = euler_lagrange(
    &ke, &pe,
    &[(&q1, &qd1), (&q2, &qd2)],
    &[&qdd1, &qdd2],
);
println!("τ₁ = {}", eqs[0]);
println!("τ₂ = {}", eqs[1]);
```

Now extract the dynamic components:

```rust
// Mass matrix M(q): M_ij = ∂²T/∂q̇ᵢ∂q̇ⱼ
let mm = mass_matrix(&ke, &[&qd1, &qd2]);
println!("M[0,0] = {}", mm.get(0, 0));
println!("M[0,1] = {}", mm.get(0, 1));
println!("M[1,0] = {}", mm.get(1, 0));
println!("M[1,1] = {}", mm.get(1, 1));

// The mass matrix must be symmetric
assert!(mm.is_symmetric());

// Coriolis matrix C(q, q̇) via Christoffel symbols
let cor = coriolis_matrix(&mm, &[&q1, &q2], &[&qd1, &qd2]);
println!("C[0,0] = {}", cor.get(0, 0));
println!("C[0,1] = {}", cor.get(0, 1));

// Gravity vector g(q) = ∂V/∂q
let gv = gravity_vector(&pe, &[&q1, &q2]);
println!("g₁ = {}", gv[0]);
println!("g₂ = {}", gv[1]);
```

### Convenience: `manipulator_equation()`

Get all three components in one call:

```rust
let (mass, cor, grav) = manipulator_equation(
    &ke, &pe,
    &[&q1, &q2],
    &[&qd1, &qd2],
);
println!("M shape: {:?}", mass.shape());  // (2, 2)
println!("C shape: {:?}", cor.shape());   // (2, 2)
println!("g length: {}", grav.len());     // 2
```

### With Units — Dimension-Checked Dynamics

The Lagrangian dynamics API also supports typed quantities. This ensures that
your kinetic energy is actually an `Energy`, your potential energy is `Energy`,
and differentiation with respect to angular velocity yields angular momentum:

```rust
use symplex::units::*;

// Typed symbols
let m = Mass::symbol("m");
let l = Length::symbol("L");
let g_accel = Acceleration::symbol("g");
let q = Angle::symbol("q");
let qd = AngularVelocity::symbol("qd");

// Kinetic energy — dimension checked
let ke = Energy::from_ex(expr!(1/2 * m * l^2 * qd^2));

// Potential energy — dimension checked
let pe = Energy::from_ex(expr!(m * g * l * (1 - cos(q))));

// ∂L/∂θ̇ → angular momentum (typed DiffWrt)
let theta_dot_var = AngularVelocity::symbol("qd");
let p: AngularMomentum = lagrangian.diff_wrt(&theta_dot_var);
// If any dimension is wrong, this is a compile error — not a runtime surprise.
```

The typed wrappers call the same underlying `euler_lagrange()` machinery.
They add zero runtime cost — the dimension tags exist only at compile time.

### Total Time Derivative

The `total_time_derivative` function computes $d/dt$ of an expression by applying the chain rule through generalized coordinates:

```rust
let dq1_dt = total_time_derivative(
    &q1,
    &[(&q1, &qd1), (&q2, &qd2)],
    &[&qdd1, &qdd2],
);
println!("d/dt(q1) = {dq1_dt}");
// d/dt(q1) = qd1
```

This is useful for verifying energy conservation or deriving constraint forces.

### Numerical Evaluation

Verify the dynamics at a known configuration:

```rust
// At rest, hanging straight down: q=(0,0), qd=(0,0), qdd=(1,0)
// m1=m2=1, L1=L2=1, g=10
// (eqs[] comes from the euler_lagrange() call in the double-pendulum section above)
let tau1 = eqs[0].eval_f64_with(&[
    (&m1, 1), (&m2, 1), (&l1, 1), (&l2, 1), (&g, 10),
    (&q1, 0), (&q2, 0), (&qd1, 0), (&qd2, 0), (&qdd1, 1), (&qdd2, 0),
]);
println!("τ₁ = {:.4}", tau1.unwrap_or(f64::NAN));
// τ₁ = 2.0000  (M[0,0]·1 + 0 — gravity vanishes at q=0)
```

## Step 6: Generate Code for Dynamics

The real power is combining symbolic dynamics with code generation. Derive the mass matrix symbolically, then emit optimized Rust code:

```rust
let mass_code = mm
    .to_rust_fn("mass_matrix", &["q1", "q2"])
    .expect("codegen failed");
println!("{mass_code}");
```

This generates a function `pub fn mass_matrix(q1: f64, q2: f64) -> [f64; 4]` with CSE across all four entries. The generated code runs in your real-time control loop at full compiled speed.

You can do the same for the Jacobian from Step 4:

```rust
let jac_code = jac
    .to_rust_fn("jacobian", &["theta1", "theta2"])
    .expect("codegen failed");
println!("{jac_code}");
```

### The Full Pipeline, Summarized

```
DH Parameters
    │
    ▼
fk_position()          → symbolic (px, py, pz)
    │
    ▼
jacobian()             → symbolic 2×3 Jacobian matrix
    │
    ▼
euler_lagrange()       → symbolic equations of motion
mass_matrix()          → symbolic M(q)
coriolis_matrix()      → symbolic C(q, q̇)
gravity_vector()       → symbolic g(q)
    │
    ▼
to_rust_fn()           → optimized Rust code with CSE
    │
    ▼
compile & deploy       → runs at >1 MHz in your control loop
```

This entire pipeline runs once at design time (or in `build.rs`). The output is a set of pure Rust functions that have no runtime dependency on symplex.

## Aspirational Features

The following capabilities are planned but not yet implemented:

```rust
// PLANNED: n-DOF inverse kinematics
// For 6R manipulators with a spherical wrist, Pieper's method
// decomposes IK into a position subproblem (3 DOF) and an
// orientation subproblem (3 DOF), each solvable in closed form.
// let solutions = inverse_kinematics_6r(&dh_params, &target_pose);

// PLANNED: Kane's method for dynamics
// An alternative to Euler-Lagrange that avoids computing the
// full Lagrangian — often more efficient for complex mechanisms.
// let (fr, fr_star) = kanes_equations(&bodies, &joints);

// PLANNED: Trajectory planning
// Generate minimum-jerk or time-optimal trajectories
// between joint configurations.
// let traj = min_jerk_trajectory(&q_start, &q_end, duration);

// PLANNED: 3D robot visualization
// Export the kinematic chain as a 3D scene for visual verification.
// arm.visualize_svg("robot.svg", &joint_config);
```

## Summary

| Operation | API | Notes |
|-----------|-----|-------|
| Forward kinematics | `fk_position(&dh)` | Returns `(px, py, pz)` |
| Full FK chain | `fk_chain(&dh)` | 4×4 homogeneous matrix |
| FK rotation | `fk_rotation(&dh)` | 3×3 rotation submatrix |
| Jacobian | `jacobian(&[&f1, &f2], &[&θ1, &θ2])` | Symbolic partial derivatives |
| 2-DOF IK | `inverse_kinematics_2dof(l1, l2, tx, ty)` | Gröbner-basis exact solver |
| Euler-Lagrange | `euler_lagrange(&ke, &pe, &coords, &accels)` | Equations of motion |
| Mass matrix | `mass_matrix(&ke, &velocities)` | $M(q)$ |
| Coriolis matrix | `coriolis_matrix(&M, &coords, &vels)` | $C(q, \dot{q})$ |
| Gravity vector | `gravity_vector(&pe, &coords)` | $g(q)$ |
| Total time derivative | `total_time_derivative(&expr, &coords, &accels)` | Chain rule $d/dt$ |
| Code generation | `jac.to_rust_fn("name", &["θ1", "θ2"])` | Optimized Rust with CSE |
| Embedded codegen | `jac.to_rust_fn_with_options(...)` | `f32`, `no_std` support |

## What's Next

Need to solve the differential equations that arise in dynamics analysis? [Chapter 15: ODE Solving](15-ode-solving.md) covers symplex's symbolic ODE solver — from simple separable equations to second-order systems.

---

*[← Chapter 13: Control Systems](13-control-systems.md) | [Back to Table of Contents](index.md) | [Chapter 15: ODE Solving →](15-ode-solving.md)*