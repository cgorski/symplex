# PID Controller Design

## Problem

You have a DC motor modeled as a second-order transfer function. You need to:

1. Design a PID controller with symbolic gains
2. Derive the closed-loop characteristic polynomial
3. Verify stability using the Routh-Hurwitz criterion
4. Choose specific gains and confirm all closed-loop poles are in the left half-plane
5. Generate optimized Rust code for the controller update function

## Background

A PID controller computes the control signal as:

```text
u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de/dt
```

where `e(t)` is the tracking error. In the Laplace domain, the controller transfer function is:

```text
C(s) = Kp + Ki/s + Kd·s = (Kd·s² + Kp·s + Ki) / s
```

The closed-loop system is stable when all roots of the characteristic polynomial have negative real parts. For a third-order polynomial `s³ + a₂s² + a₁s + a₀`, the Routh-Hurwitz conditions are:

- `a₂ > 0`
- `a₀ > 0`
- `a₂·a₁ > a₀`

## Solution

The complete solution is in `examples/pid_controller.rs`. Run it with:

```sh
cargo run --example pid_controller
```

### Setup

The plant is a normalized DC motor model with moment of inertia J=1, damping b=10, and torque constant K=20:

```rust
use symplex::prelude::*;

let ctx = Context::new();
symplex::syms!(ctx; s, Kp, Ki, Kd);

// Plant: G(s) = 20 / (s² + 10s)
// Closed-loop characteristic polynomial with PID:
let char_poly = expr!(ctx,
    s^3 + (10 + 20*Kd)*s^2 + 20*Kp*s + 20*Ki
);
```

### Symbolic Stability Analysis

The Routh-Hurwitz conditions are derived directly from the coefficients:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# symplex::syms!(ctx; s, Kp, Ki, Kd);
let a2 = expr!(ctx, 10 + 20*Kd);
let a1 = expr!(ctx, 20*Kp);
let a0 = expr!(ctx, 20*Ki);

// Stability requires: a2 > 0, a0 > 0, a2·a1 > a0
let routh_product = (&a2 * &a1).expand();
// → 400*Kd*Kp + 200*Kp
```

### Gain Selection and Verification

Substituting Kp=5, Ki=2, Kd=0.5 gives `P(s) = s³ + 20s² + 100s + 40`. symplex finds the three closed-loop poles numerically and confirms all have negative real parts:

```text
Closed-loop poles:
  p1 = -0.4374 ✓
  p2 = -11.8382 ✓
  p3 = -7.7244 ✓

Stability: STABLE — all poles in left half-plane
```

The Routh conditions are also verified: `a₂·a₁ = 2000 > a₀ = 40`.

### Code Generation

The PID update equation with the chosen gains is compiled to an optimized Rust function:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
symplex::syms!(ctx; error, integral, derivative);
let pid_output = &ctx.rational(5, 1) * &error
    + &ctx.rational(2, 1) * &integral
    + &ctx.rational(1, 2) * &derivative;

let code = pid_output.eval().to_rust_fn(
    "pid_update", &["error", "integral", "derivative"]
).unwrap();

```

This produces:

```rust
#[must_use]
pub fn pid_update(error: f64, integral: f64, derivative: f64) -> f64 {
    5_f64.mul_add(error, 2_f64.mul_add(integral, (0.5_f64 * derivative)))
}
```

The generated function can be dropped directly into an embedded control loop. It uses `mul_add` for numerical stability and contains no allocations, branches, or function calls beyond basic arithmetic.

## Key symplex Features Used

- `expr!` macro for building polynomial expressions with symbolic gains
- `.expand()` for multiplying out the Routh product
- `.subs()` and `.eval()` for substituting concrete gain values
- `.solve()` for finding closed-loop poles (cubic equation)
- `.eval_f64()` and `.eval_complex64()` for numerical pole evaluation
- `.to_rust_fn()` for generating deployable Rust code
- `.compile()` for creating a callable closure for verification