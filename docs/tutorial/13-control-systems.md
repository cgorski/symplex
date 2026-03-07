# Chapter 13: Control Systems

Symplex provides a complete symbolic control systems toolkit: state-space models, transfer functions, stability analysis, pole placement, discretization, and Laplace transforms. If you've used MATLAB's Control System Toolbox, you'll feel right at home — except everything is exact, and the results are Rust code you can deploy.

This chapter walks through a realistic workflow: model a physical plant, analyze it, design a controller, and discretize for digital implementation.

## The Plant: A Mass-Spring-Damper

Every controls course starts here. A mass $m$ on a spring (stiffness $k$) with a viscous damper (coefficient $c$), driven by force $F$:

$$m\ddot{x} + c\dot{x} + kx = F$$

With $m = 1$, $c = 3$, $k = 4$, we get:

$$\ddot{x} + 3\dot{x} + 4x = F$$

Let's build the state-space representation and analyze it end-to-end.

## State-Space Construction

Choose state variables $x_1 = x$ (position) and $x_2 = \dot{x}$ (velocity). The state equations become:

$$\dot{x}_1 = x_2$$
$$\dot{x}_2 = -4x_1 - 3x_2 + F$$

With output $y = x_1$ (position measurement):

```rust
use symplex::prelude::*;
use symplex::control::{StateSpace, TransferFunction, is_routh_stable};
use symplex::vars;

vars!(s);

// State-space matrices: ẋ = Ax + Bu, y = Cx + Du
let a = matrix![[0, 1], [-4, -3]];
let b = matrix![[0], [1]];
let c = matrix![[1, 0]];
let d = matrix![[0]];

let sys = StateSpace::new(a.clone(), b.clone(), c.clone(), d.clone());

println!("A = {a}");
println!("B = {b}");
println!("C = {c}");
println!("D = {d}");
println!(
    "Dimensions: {} states, {} inputs, {} outputs",
    sys.num_states(),
    sys.num_inputs(),
    sys.num_outputs()
);
// Dimensions: 2 states, 1 inputs, 1 outputs
```

The `StateSpace::new` constructor validates dimensional consistency — `A` must be square, `B` must have as many rows as `A`, and so on. If you get the sizes wrong, it panics with a clear message.

> ### 🔬 With Units — Verifying Matrix Entry Dimensions
>
> The state matrix entries have specific physical dimensions that encode the
> physics of the plant. With symplex's dimensional analysis you can verify these
> at compile time:
>
> ```rust
> // The state matrix entries have specific physical dimensions:
> // A[0,1] = 1/s (converts velocity to position rate)
> // A[1,0] = -k/m = -4/s² (spring stiffness / mass)
> // A[1,1] = -c/m = -3/s (damping / mass)
> //
> // Verify with const_assert_dim!:
> symplex::const_assert_dim!(
>     ConstDim::STIFFNESS.div(ConstDim::MASS),
>     ConstDim::new(0, 0, -2, 0, 0, 0, 0),
>     "k/m must have dimension T⁻² (angular frequency squared)"
> );
> symplex::const_assert_dim!(
>     ConstDim::DAMPING.div(ConstDim::MASS),
>     ConstDim::new(0, 0, -1, 0, 0, 0, 0),
>     "c/m must have dimension T⁻¹ (inverse time)"
> );
> ```
>
> You can also build the force equation from typed symbols, so that dimensional
> mismatches (e.g., adding a force to a velocity) are caught at compile time:
>
> ```rust
> use symplex::units::*;
>
> // With units:
> let mass = Mass::symbol("m");
> let stiffness = Stiffness::symbol("k");
> let damping = Damping::symbol("c");
> let x = Length::symbol("x");
> let v = Velocity::symbol("v");
>
> let f_spring: Force = -(&stiffness * &x);
> let f_damp: Force = -(&damping * &v);
> // f_spring + f_damp is a Force — adding a Length here would be a compile error
> ```
>
> See [Chapter 21: Units & Dimensional Analysis](21-units.md) for the full
> reference on typed quantities.

## Poles and Stability

The poles of the system are the eigenvalues of **A**. They tell you everything about the natural response.

```rust
// Characteristic polynomial: det(sI - A)
let char_poly = sys.char_poly(&s);
println!("Characteristic polynomial: {char_poly}");
// s^2 + 3*s + 4

// Poles (eigenvalues of A)
let poles = sys.poles(&s);
for (i, p) in poles.iter().enumerate() {
    println!("  p{} = {p}", i + 1);
}
// p1 = -3/2 + sqrt(7)*i/2    (complex conjugate pair)
// p2 = -3/2 - sqrt(7)*i/2

// Stability check: are all poles in the left half-plane?
match sys.is_stable() {
    Some(true) => println!("System is stable"),
    Some(false) => println!("System is UNSTABLE"),
    None => println!("Stability undetermined"),
}
// System is stable
```

Both poles have negative real part ($-3/2$), so the system is stable. The imaginary part ($\pm\sqrt{7}/2$) means it oscillates — this is an underdamped second-order system. No floating-point approximation needed to reach that conclusion.

## Controllability and Observability

Before designing a controller, verify that the system is controllable (we can steer the state anywhere) and observable (we can reconstruct the state from output measurements):

```rust
println!("Controllable: {}", sys.is_controllable());
// Controllable: true

println!("Observable:   {}", sys.is_observable());
// Observable: true

// Controllability matrix: [B, AB]
let ctrb = sys.controllability_matrix();
println!("Controllability matrix: {ctrb}");
println!("  rank = {}", ctrb.rank());
// rank = 2 (full rank → controllable)

// Observability matrix: [C; CA]
let obsv = sys.observability_matrix();
println!("Observability matrix: {obsv}");
println!("  rank = {}", obsv.rank());
// rank = 2 (full rank → observable)
```

A rank-deficient controllability matrix means some states can't be influenced by the input. A rank-deficient observability matrix means some states are invisible in the output. Neither is the case here — good news for our controller design.

## Transfer Functions

The transfer function is the Laplace-domain input-output relationship $G(s) = C(sI - A)^{-1}B + D$:

```rust
// Construct directly from numerator/denominator coefficients
// G(s) = 1 / (s² + 3s + 4)
// Coefficients are in ascending power order: [constant, s, s², ...]
let tf = TransferFunction::from_coeffs(&[1], &[4, 3, 1], &s);
println!("G(s) = {tf}");

// DC gain: G(0) — the steady-state output for a unit step input
println!("DC gain: {}", tf.dc_gain());
// DC gain: 1/4

// Poles and zeros of the transfer function
let tf_poles = tf.poles();
println!("Poles: {:?}", tf_poles.iter().map(|p| format!("{p}")).collect::<Vec<_>>());

let tf_zeros = tf.zeros();
println!("Zeros: {:?}", tf_zeros.iter().map(|z| format!("{z}")).collect::<Vec<_>>());
// Zeros: []  (no finite zeros for this system)
```

The DC gain of $1/4$ means that a unit step force produces a steady-state displacement of $0.25$ meters. That's the ratio $1/k$ — exactly what you'd expect from static analysis.

## Transfer Function Algebra

Series, parallel, and feedback connections are the building blocks of block diagram reduction:

```rust
let g1 = TransferFunction::from_coeffs(&[1], &[1, 1], &s); // 1/(s+1)
let g2 = TransferFunction::from_coeffs(&[1], &[2, 1], &s); // 1/(s+2)

// Series: G1(s) · G2(s)
let series = g1.series(&g2);
println!("Series:   {series}");
// 1 / ((s+1)(s+2))

// Parallel: G1(s) + G2(s)
let parallel = g1.parallel(&g2);
println!("Parallel: {parallel}");

// Unity feedback: G1 / (1 + G1)
let fb = g1.feedback();
println!("Unity feedback: {fb}");

// Feedback with sensor H(s): G1 / (1 + G1·H)
let fb_with = g1.feedback_with(&g2);
println!("Feedback with sensor: {fb_with}");
```

These operations keep everything symbolic — no polynomial truncation, no numerical cancellation issues.

## Routh-Hurwitz Stability Criterion

When you have a characteristic polynomial but don't need the actual pole locations, the Routh-Hurwitz criterion tells you whether all roots are in the left half-plane without solving for them:

```rust
use symplex::control::{is_routh_stable, routh_array};

// s² + 3s + 4 — our mass-spring-damper
let coeffs = [symplex::int(1), symplex::int(3), symplex::int(4)];
match is_routh_stable(&coeffs) {
    Some(true)  => println!("Routh stable: yes"),
    Some(false) => println!("Routh stable: no"),
    None        => println!("Undetermined"),
}
// Routh stable: yes

// A more interesting case: s³ + 2s² + 3s + 4
let coeffs3 = [symplex::int(1), symplex::int(2), symplex::int(3), symplex::int(4)];
match is_routh_stable(&coeffs3) {
    Some(true)  => println!("Routh stable: yes"),
    Some(false) => println!("Routh stable: no"),
    None        => println!("Undetermined"),
}
// Routh stable: yes
// (First column of the Routh array is [1, 2, 1, 4] — all positive, so
//  all three roots have negative real parts and the polynomial is stable.)

// Inspect the full Routh array
let routh = routh_array(&coeffs3);
for (i, row) in routh.iter().enumerate() {
    let entries: Vec<String> = row.iter().map(|e| format!("{e}")).collect();
    println!("  Row {i}: [{}]", entries.join(", "));
}
```

The Routh array is computed with exact rational arithmetic — no round-off that might flip a marginal stability result.

```rust
// Clearly unstable: s³ + s² - 2s + 1 (negative coefficient)
let unstable = [symplex::int(1), symplex::int(1), symplex::int(-2), symplex::int(1)];
match is_routh_stable(&unstable) {
    Some(false) => println!("Unstable: sign changes in first column"),
    _ => {}
}
// Unstable: sign changes in first column
// (The first column is [1, 1, -3, 1] — two sign changes, meaning two
//  roots in the right half-plane.)
```

## Ackermann Pole Placement

Now for controller design. The original system has poles at $-3/2 \pm j\sqrt{7}/2$ — it's stable but slow and oscillatory. We want faster, well-damped poles. Ackermann's formula gives us the state-feedback gain $K$ such that the closed-loop system $\dot{x} = (A - BK)x$ has the desired poles:

```rust
// Place poles at s = -5 and s = -6 (faster, no oscillation)
let desired_poles = [symplex::int(-5), symplex::int(-6)];

match sys.ackermann(&desired_poles) {
    Some(k) => {
        println!("Feedback gain K = {k}");

        // Verify: eigenvalues of (A - BK) should match desired poles
        let bk = b.matmul(&k);
        let a_cl = a.sub(&bk);
        let cl_poles = a_cl.eigenvals(&s);
        println!("Closed-loop poles: {:?}",
            cl_poles.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
        );
        // Should print: ["-5", "-6"]
    }
    None => println!("Pole placement failed"),
}
```

The gain vector $K$ is computed exactly. No numerical eigenvalue solver involved — Ackermann's formula uses the characteristic polynomial evaluated at the $A$ matrix, which symplex computes symbolically.

### Complex Pole Placement

You can also place complex conjugate poles for a specific damping ratio and natural frequency:

```rust
// Place poles at s = -2 ± 3j (damped oscillation)
let i_unit = symplex::i_unit();
let p1 = &symplex::int(-2) + &(&i_unit * 3);
let p2 = &symplex::int(-2) - &(&i_unit * 3);

match sys.ackermann(&[p1, p2]) {
    Some(k) => println!("Gain for complex poles: K = {k}"),
    None => println!("Ackermann failed for complex poles"),
}
```

## ZOH Discretization

Real controllers run on digital hardware. Zero-order hold (ZOH) discretization converts your continuous-time model to a discrete-time equivalent suitable for implementation at a fixed sample rate:

```rust
// Discretize with sample time dt = 0.01 s (100 Hz control loop)
let dt = symplex::rational(1, 100);

// The second argument is the Taylor series order for the matrix exponential
let discrete = sys.discretize_zoh(&dt, 4);

println!("Discrete A (Ad): {}", discrete.a);
println!("Discrete B (Bd): {}", discrete.b);
println!(
    "States: {}, Inputs: {}, Outputs: {}",
    discrete.num_states(),
    discrete.num_inputs(),
    discrete.num_outputs()
);

// Discrete-time stability: all eigenvalues inside unit circle |z| < 1
let disc_poles = discrete.poles(&s);
println!("Discrete poles: {:?}",
    disc_poles.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
);

match discrete.is_stable() {
    Some(true) => println!("Discrete system stable: yes"),
    Some(false) => println!("Discrete system stable: no"),
    None => println!("Discrete stability undetermined"),
}
```

The ZOH discretization uses a Taylor series approximation of $e^{AT_s}$. The 4th-order approximation is typically sufficient for small sample times. For the exact result, increase the order — symplex keeps everything rational, so there's no numerical blow-up.

## Laplace Transforms

Symplex's Laplace transform engine connects time-domain analysis to transfer-function analysis. This is how you go from a differential equation to a transfer function and back:

### Forward Transform

```rust
vars!(t, s);

// Common transform pairs
println!("L{{1}} = {}", symplex::int(1).laplace(&t, &s).unwrap());
// 1/s

println!("L{{t}} = {}", t.laplace(&t, &s).unwrap());
// s^(-2)

println!("L{{exp(-3t)}} = {}", (-&t * 3).exp().laplace(&t, &s).unwrap());
// 1/(s + 3)

println!("L{{sin(2t)}} = {}", (&t * 2).sin().laplace(&t, &s).unwrap());
// 2/(s^2 + 4)

println!("L{{cos(2t)}} = {}", (&t * 2).cos().laplace(&t, &s).unwrap());
// s/(s^2 + 4)
```

### Inverse Transform and Impulse Response

The inverse Laplace transform uses partial fraction decomposition followed by table lookup — the same technique you'd use by hand, but exact:

```rust
// Impulse response of our mass-spring-damper
// G(s) = 1 / (s² + 3s + 4)
let gs = 1 / &(expr!(s ^ 2 + 3 * s + 4));

match gs.inverse_laplace(&s, &t) {
    Ok(ht) => println!("Impulse response h(t) = {ht}"),
    Err(e) => println!("Inverse Laplace failed: {e}"),
}
```

### Frequency Response Evaluation

Evaluate the transfer function at $s = j\omega$ to get the frequency response:

```rust
let i_unit = symplex::i_unit();
let gs = 1 / &(expr!(s ^ 2 + 3 * s + 4));

for omega in [1i64, 2, 5, 10] {
    let jw = &i_unit * omega;
    let g_jw = gs.subs(&s, &jw).eval();
    println!("G(j·{omega}) = {g_jw}");
}
```

Each evaluation is exact in complex arithmetic — you get the real and imaginary parts as exact rationals times algebraic numbers.

## Complete Workflow: Design and Deploy

Here's the full pipeline a controls engineer would follow, from physics to deployable code:

```rust
use symplex::prelude::*;
use symplex::control::{StateSpace, TransferFunction, is_routh_stable};
use symplex::vars;

fn main() {
    vars!(s);

    // 1. Model the plant
    let a = matrix![[0, 1], [-4, -3]];
    let b = matrix![[0], [1]];
    let c = matrix![[1, 0]];
    let d = matrix![[0]];
    let sys = StateSpace::new(a.clone(), b.clone(), c.clone(), d.clone());

    // 2. Verify it's stabilizable
    assert!(sys.is_controllable(), "must be controllable for pole placement");
    assert!(matches!(sys.is_stable(), Some(true)), "open-loop should be stable");

    // 3. Design: place poles for faster response
    let desired = [symplex::int(-5), symplex::int(-6)];
    let k = sys.ackermann(&desired).expect("pole placement succeeded");
    println!("Feedback gain K = {k}");

    // 4. Verify closed-loop stability via Routh
    let a_cl = a.sub(&b.matmul(&k));
    let cl_char = a_cl.char_poly(&s);
    println!("Closed-loop characteristic poly: {cl_char}");

    // 5. Discretize for 100 Hz implementation
    let dt = symplex::rational(1, 100);
    let discrete = sys.discretize_zoh(&dt, 4);
    assert!(matches!(discrete.is_stable(), Some(true)));

    // 6. Generate code for the discrete-time controller
    let code = k.to_rust_fn("feedback_gain", &["x1", "x2"])
        .expect("codegen succeeded");
    println!("{code}");
}
```

## Aspirational Features

The following features are planned but not yet implemented:

```rust
// PLANNED: Block diagram algebra types
// let loop_tf = Series(controller, plant);
// let closed = Feedback(loop_tf, sensor);
// let overall = Series(prefilter, closed);

// PLANNED: Multi-input multi-output transfer functions
// let mimo = MIMOTransferFunction::from_matrix(&tf_matrix, &s);

// PLANNED: Bode plot as SVG
// tf.bode_svg("bode.svg", 0.01..1000.0);

// PLANNED: Step response plot
// tf.step_response_svg("step.svg", 0.0..5.0);

// PLANNED: Nyquist plot
// tf.nyquist_svg("nyquist.svg");
```

## Summary

| Operation | API | Notes |
|-----------|-----|-------|
| State-space model | `StateSpace::new(a, b, c, d)` | Validates dimensions |
| Transfer function | `TransferFunction::from_coeffs(&num, &den, &s)` | Ascending power order |
| Poles / zeros | `sys.poles(&s)` / `tf.zeros()` | Exact symbolic |
| Stability | `sys.is_stable()` | Returns `Option<bool>` |
| Controllability | `sys.is_controllable()` | Rank of $[B, AB, \ldots]$ |
| Observability | `sys.is_observable()` | Rank of $[C; CA; \ldots]$ |
| DC gain | `tf.dc_gain()` | $G(0)$ |
| Routh-Hurwitz | `is_routh_stable(&coeffs)` | No pole computation needed |
| Pole placement | `sys.ackermann(&poles)` | Full state feedback |
| ZOH discretization | `sys.discretize_zoh(&dt, order)` | Taylor-based $e^{AT_s}$ |
| Laplace transform | `expr.laplace(&t, &s)` | Table + linearity |
| Inverse Laplace | `expr.inverse_laplace(&s, &t)` | Partial fractions + table |
| Series connection | `tf1.series(&tf2)` | $G_1 \cdot G_2$ |
| Parallel connection | `tf1.parallel(&tf2)` | $G_1 + G_2$ |
| Feedback | `tf.feedback()` / `tf.feedback_with(&h)` | $G/(1+GH)$ |

## What's Next

If you're building robot controllers, head to [Chapter 14: Robotics](14-robotics.md) to derive kinematics and dynamics symbolically — and generate the fast numerical code your control loop needs.

---

*[← Chapter 12: Quaternions](12-quaternions.md) | [Back to Table of Contents](index.md) | [Chapter 14: Robotics →](14-robotics.md)*