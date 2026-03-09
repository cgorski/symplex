# Chapter 8: Code Generation

**This is the killer feature.** Everything else in symplex — differentiation, integration, solving, simplification — exists in other CAS tools. Code generation is what makes symplex unique: you derive symbolically, then emit optimized Rust code that runs at full compiled speed.

The workflow is: **derive once, generate once, run forever.**

## The Vision

In robotics, controls, and physics, there's a recurring pattern:

1. **Derive** the math symbolically — forward kinematics from DH parameters, equations of motion from Lagrangians, Jacobians from position expressions.
2. **Simplify** and verify the symbolic result.
3. **Convert** the symbolic expression to fast numerical code.
4. **Run** that code in a real-time loop at thousands of Hz.

Traditional approaches either do step 1–2 in Python/MATLAB and hand-translate to C/Rust (error-prone, tedious), or evaluate the symbolic tree at runtime (slow). Symplex collapses steps 2–3 into a single function call: `.to_rust_fn()`.

## Simple Example: A Polynomial

Let's start with the simplest possible case:

```rust
use symplex::prelude::*;

let ctx = Context::new();
fn main() {
    syms!(ctx; x);

    let f = expr!(x^3 - 3*x^2 + 2*x + 7);
    let code = f.to_rust_fn("cubic", &["x"]).unwrap();
    println!("{code}");
}
```

This prints something like:

```rust
#[must_use]
pub fn cubic(x: f64) -> f64 {
    x * x * x - 3.0 * x * x + 2.0 * x + 7.0
}
```

That's a complete, compilable Rust function. No runtime overhead. No expression-tree traversal. Just `f64` arithmetic that LLVM can optimize further.

### What the Generated Code Includes

Every generated function has:

- A `pub fn` signature with named `f64` parameters
- A `#[must_use]` annotation (by default) so callers don't accidentally discard the return value
- CSE (common subexpression elimination) — shared subexpressions are computed once and stored in `let` bindings
- Clean decimal formatting — `3.0` not `3.0000000000000000`
- Proper subtraction — `x - 3.0` not `x + (-3.0)`

## Scalar Code Generation

### `to_rust_fn()`

The basic code generation function:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let f = x.sin().powi(2) + x.cos().powi(2);
let code = f.to_rust_fn("trig_identity", &["x"]).unwrap();
println!("{code}");
// The CSE may extract sin(x) and cos(x) as temporaries
```

Arguments:
- `name` — the function name
- `args` — slice of parameter names, determining positional mapping

The variable names you pass to `args` must match the symbol names in the expression. If your expression uses `ctx.var("theta1")`, then `args` should contain `"theta1"`.

### `to_rust_fn_with_options()`

For fine-grained control:

```rust
use symplex::prelude::*;
use symplex::matrix::CodegenOptions;

let ctx = Context::new();
syms!(ctx; x);

let f = expr!(x^2 + sin(x));
let opts = CodegenOptions::default();
let code = f.to_rust_fn_with_options("my_func", &["x"], &opts).unwrap();
println!("{code}");
```

## CodegenOptions

The `CodegenOptions` struct controls every aspect of code generation:

```rust
use symplex::matrix::{CodegenOptions, MathBackend, Precision};

// Default: std math, f64, no inline, must_use, CSE enabled
let default = CodegenOptions::default();

// For no_std embedded targets
let no_std = CodegenOptions::no_std();

// For Cortex-M4F and similar f32 targets
let embedded = CodegenOptions::embedded_f32();

// Full manual control
let custom = CodegenOptions {
    math_backend: MathBackend::Std,
    precision: Precision::F64,
    inline: true,
    must_use: true,
    cse: true,
};
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `math_backend` | `MathBackend` | `Std` | How math functions are dispatched |
| `precision` | `Precision` | `F64` | `f64` or `f32` output |
| `inline` | `bool` | `false` | Add `#[inline]` annotation |
| `must_use` | `bool` | `true` | Add `#[must_use]` annotation |
| `cse` | `bool` | `true` | Run common subexpression elimination |

### MathBackend Variants

| Variant | Description | Use When |
|---------|-------------|----------|
| `MathBackend::Std` | Uses `f64::sin()`, `f64::cos()`, etc. | Default — standard Rust targets |
| `MathBackend::Libm` | Uses `libm::sin()`, `libm::cos()`, etc. | `no_std` with the `libm` crate |
| `MathBackend::CfgGated` | Emits a helper module with `#[cfg]` gates | Portable code that works with or without `std` |

### Precision Variants

| Variant | Type | Constants | Use When |
|---------|------|-----------|----------|
| `Precision::F64` | `f64` | `std::f64::consts::PI` | Default — double precision |
| `Precision::F32` | `f32` | `std::f32::consts::PI` | Embedded, GPU, memory-constrained |

### Preset Configurations

**`CodegenOptions::no_std()`** — For `no_std` environments:
- `CfgGated` math backend
- `#[inline]` enabled
- `f64` precision
- CSE enabled

```rust
use symplex::prelude::*;
use symplex::matrix::CodegenOptions;

let ctx = Context::new();
syms!(ctx; x);

let f = x.sin() + x.cos();
let code = f.to_rust_fn_with_options("trig_sum", &["x"], &CodegenOptions::no_std()).unwrap();
println!("{code}");
// Generates code with cfg-gated math module
```

**`CodegenOptions::embedded_f32()`** — For Cortex-M4F and similar:
- `CfgGated` math backend
- `#[inline]` enabled
- `f32` precision
- CSE enabled

```rust
use symplex::prelude::*;
use symplex::matrix::CodegenOptions;

let ctx = Context::new();
syms!(ctx; x);

let f = x.sin() + x.cos();
let code = f.to_rust_fn_with_options("trig_sum_f32", &["x"], &CodegenOptions::embedded_f32()).unwrap();
println!("{code}");
// Uses f32 types, f32 constants, cfg-gated math
```

## Common Subexpression Elimination (CSE)

CSE is the optimization that makes generated code efficient. When the same subexpression appears multiple times, CSE extracts it into a temporary variable computed once:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// sin(x) appears in both terms
let f = &x.sin().powi(2) + &x.sin();
let code = f.to_rust_fn("with_cse", &["x"]).unwrap();
println!("{code}");
```

The generated code looks something like:

```rust
#[must_use]
pub fn with_cse(x: f64) -> f64 {
    let __cse_0 = x.sin();
    __cse_0 * __cse_0 + __cse_0
}
```

Without CSE, `sin(x)` would be computed twice. For robotics Jacobians with dozens of shared trig terms, CSE can reduce computation by 50% or more.

### Disabling CSE

If you want the raw (un-optimized) output for debugging:

```rust
use symplex::prelude::*;
use symplex::matrix::CodegenOptions;

let ctx = Context::new();
syms!(ctx; x);

let f = &x.sin().powi(2) + &x.sin();
let opts = CodegenOptions { cse: false, ..Default::default() };
let code = f.to_rust_fn_with_options("no_cse", &["x"], &opts).unwrap();
println!("{code}");
```

### Manual CSE

You can also run CSE manually to inspect the extracted subexpressions:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let f = &x.sin().powi(2) + &x.sin();
let (bindings, result) = f.cse();

println!("Bindings:");
for (name, value) in &bindings {
    println!("  {name} = {value}");
}
println!("Result: {result}");
```

## Matrix Code Generation

Matrix code generation is where things get really powerful. For a symbolic matrix, `.to_rust_fn()` generates a function that returns a flat array `[f64; R*C]` in row-major order.

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let m = matrix![[x.sin(), x.cos()], [-x.cos(), x.sin()]];
let code = m.to_rust_fn("rotation_2d", &["x"]).expect("codegen");
println!("{code}");
```

Output (approximately):

```rust
#[must_use]
pub fn rotation_2d(x: f64) -> [f64; 4] {
    let __cse_0 = x.sin();
    let __cse_1 = x.cos();
    [
        __cse_0,
        __cse_1,
        -__cse_1,
        __cse_0,
    ]
}
```

Notice: `sin(x)` and `cos(x)` are each computed once, even though they appear in multiple matrix entries. This is **cross-entry CSE** — the CSE pass runs across all entries simultaneously, not per-entry.

### Matrix CodegenOptions

Matrix code generation accepts the same `CodegenOptions`:

```rust
use symplex::prelude::*;
use symplex::matrix::CodegenOptions;

let ctx = Context::new();
syms!(ctx; theta);

let neg_sin = -&theta.sin();
let m = Matrix::new(vec![
    vec![theta.cos(), neg_sin],
    vec![theta.sin(), theta.cos()],
]);

// Default (f64, std, CSE)
let code = m.to_rust_fn("rot", &["theta"]).unwrap();
println!("{code}");

// Embedded f32
let opts = CodegenOptions::embedded_f32();
let code = m.to_rust_fn_with_options("rot_f32", &["theta"], &opts).unwrap();
println!("{code}");
```

## The Robotics Workflow: End-to-End

Here's the complete pipeline from DH parameters to compiled code — the canonical symplex use case.

### Step 1: Define the Robot

A 3-DOF planar robot arm with link lengths L₁, L₂, L₃ and joint angles θ₁, θ₂, θ₃:

```rust
use symplex::prelude::*;
use symplex::matrix::jacobian;
use symplex::robotics::*;

let ctx = Context::new();
fn main() {
    // Joint angles
    syms!(ctx; theta1, theta2, theta3);

    // Link lengths (exact rationals — no floating-point)
    let l1 = ctx.rational(3, 10); // 0.3 m
    let l2 = ctx.rational(1, 4);  // 0.25 m
    let l3 = ctx.rational(1, 5);  // 0.2 m

    // DH parameters: (theta, d, a, alpha)
    // All alpha = 0 for a planar arm
    let zero = ctx.int(0);
    let dh: [(&Ex, &Ex, &Ex, &Ex); 3] = [
        (&theta1, &zero, &l1, &zero),
        (&theta2, &zero, &l2, &zero),
        (&theta3, &zero, &l3, &zero),
    ];

    // ... continued below
}
```

### Step 2: Compute Forward Kinematics

```rust
    // Compute FK — returns (px, py, pz) as symbolic expressions
    let (px, py, _pz) = fk_position(&dh);

    println!("End-effector X:");
    println!("  {px}");
    println!("End-effector Y:");
    println!("  {py}");
```

The FK expressions are sums of products of `sin(θ₁)`, `cos(θ₁+θ₂)`, etc. — potentially quite large for 6-DOF arms.

### Step 3: Compute the Jacobian

```rust
    // Jacobian: ∂(px, py) / ∂(θ₁, θ₂, θ₃)
    let jac = jacobian(&[&px, &py], &[&theta1, &theta2, &theta3]);
    println!("Jacobian (2×3):");
    println!("{jac}");
```

### Step 4: Generate Code

```rust
    // Generate optimized Rust for the Jacobian
    let code = jac
        .to_rust_fn("robot_jacobian", &["theta1", "theta2", "theta3"])
        .expect("codegen");

    println!("Generated code ({} bytes):", code.len());
    println!("{code}");
```

### Step 5: Verify Numerically

Always verify the generated code against the symbolic expressions:

```rust
    // Verify: evaluate both symbolically and check consistency
    let theta1_val = 0.5_f64;
    let theta2_val = 0.3_f64;
    let theta3_val = 0.1_f64;

    let px_numeric = px.eval_f64_with(&[
        (&theta1, theta1_val as i64),
        (&theta2, theta2_val as i64),
        (&theta3, theta3_val as i64),
    ]);
    println!("FK X at (0.5, 0.3, 0.1): {:?}", px_numeric);
```

In practice, you'd compile the generated code and compare its output against the symbolic evaluation to ensure they match.

### The Complete Output

For a 3-DOF planar arm, the generated Jacobian function will look something like:

```rust
#[must_use]
pub fn robot_jacobian(theta1: f64, theta2: f64, theta3: f64) -> [f64; 6] {
    let __cse_0 = theta1 + theta2;
    let __cse_1 = __cse_0 + theta3;
    let __cse_2 = __cse_0.sin();
    let __cse_3 = __cse_1.sin();
    let __cse_4 = __cse_0.cos();
    let __cse_5 = __cse_1.cos();
    [
        -0.3 * theta1.sin() - 0.25 * __cse_2 - 0.2 * __cse_3,
        -0.25 * __cse_2 - 0.2 * __cse_3,
        -0.2 * __cse_3,
        0.3 * theta1.cos() + 0.25 * __cse_4 + 0.2 * __cse_5,
        0.25 * __cse_4 + 0.2 * __cse_5,
        0.2 * __cse_5,
    ]
}
```

Key things to notice:

1. **CSE at work:** `theta1 + theta2` is computed once as `__cse_0`. Then `__cse_0 + theta3` is `__cse_1`. Their sin/cos values are each computed once.

2. **Clean output:** Decimals are `0.3`, `0.25`, `0.2` — not `0.30000000000000004`.

3. **Flat array:** The 2×3 Jacobian is returned as `[f64; 6]` in row-major order. Entry `[i*3 + j]` corresponds to `J[i][j]`.

4. **No dependencies:** The generated function uses only `f64` arithmetic and std math functions. No runtime dependency on symplex.

## Code Generation for Scalar Expressions

Not just matrices — individual expressions also benefit from codegen:

### Derivatives

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let f = expr!(x^4 - 3*x^2 + 2*x - 1);
let df = f.diff(&x);
let code = df.to_rust_fn("derivative", &["x"]).unwrap();
println!("{code}");
```

### Control-System Transfer Functions

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; s);

// Transfer function: G(s) = (s + 2) / (s² + 3s + 4)
let num = expr!(s + 2);
let den = expr!(s^2 + 3*s + 4);
let g = &num / &den;

// Generate code to evaluate G at a specific frequency
let code = g.to_rust_fn("transfer_fn", &["s"]).unwrap();
println!("{code}");
```

### Multivariate Functions

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y, z);

let f = expr!(x^2*y + y^2*z + z^2*x);
let code = f.to_rust_fn("trivariate", &["x", "y", "z"]).unwrap();
println!("{code}");
// pub fn trivariate(x: f64, y: f64, z: f64) -> f64 { ... }
```

## Constant Folding

The code generator performs constant folding — expressions with known constant arguments are evaluated at generation time:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// sin(0) = 0, cos(0) = 1, exp(0) = 1 — folded at codegen time
let f = &x + &ctx.int(0).sin(); // sin(0) = 0, so this is just x
let code = f.to_rust_fn("with_folding", &["x"]).unwrap();
println!("{code}");
// The sin(0) term is eliminated entirely
```

Constants like `π` and `e` are emitted using Rust's standard constant modules:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let f = &x * &ctx.pi();
let code = f.to_rust_fn("with_pi", &["x"]).unwrap();
println!("{code}");
// Uses std::f64::consts::PI, not a decimal approximation
```

## The `symplex-build` Pipeline Concept

The ultimate goal is a build-script workflow where symbolic derivations happen at compile time:

```rust
// In build.rs (conceptual — symplex-build is planned)
use symplex::prelude::*;
use std::fs;

fn main() {
    syms!(ctx; theta1, theta2, theta3);

    // Derive FK and Jacobian
    let zero = ctx.int(0);
    let l1 = ctx.rational(3, 10);
    let l2 = ctx.rational(1, 4);
    let l3 = ctx.rational(1, 5);

    let dh: [(&Ex, &Ex, &Ex, &Ex); 3] = [
        (&theta1, &zero, &l1, &zero),
        (&theta2, &zero, &l2, &zero),
        (&theta3, &zero, &l3, &zero),
    ];

    let (px, py, _) = symplex::robotics::fk_position(&dh);
    let jac = symplex::matrix::jacobian(
        &[&px, &py],
        &[&theta1, &theta2, &theta3]
    );

    // Generate code
    let code = jac
        .to_rust_fn("robot_jacobian", &["theta1", "theta2", "theta3"])
        .expect("codegen failed");

    // Write to OUT_DIR
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let path = format!("{out_dir}/robot_generated.rs");
    fs::write(&path, code).unwrap();

    println!("cargo:rerun-if-changed=build.rs");
}
```

Then in your main code:

```rust
// In src/robot.rs
include!(concat!(env!("OUT_DIR"), "/robot_generated.rs"));

fn main() {
    let jac = robot_jacobian(0.5, 0.3, 0.1);
    // jac is [f64; 6], ready to use in your control loop
}
```

The symbolic math runs **once** at build time. Your final binary contains only the optimized numerical function — no dependency on symplex at runtime.

## Annotated Walkthrough of Generated Code

Let's trace through a realistic example in detail. Consider a 2-DOF rotation matrix:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; a, b);

let m = matrix![
    [a.cos() * b.cos(), -(a.sin())],
    [a.sin() * b.cos(), a.cos()]
];

let code = m.to_rust_fn("rot_ab", &["a", "b"]).unwrap();
```

The output will resemble:

```rust
#[must_use]
pub fn rot_ab(a: f64, b: f64) -> [f64; 4] {
    //  ┌─ CSE: extract shared subexpressions
    let __cse_0 = a.cos();   // cos(a) — used in entries [0,0] and [1,1]
    let __cse_1 = a.sin();   // sin(a) — used in entries [0,1] and [1,0]
    let __cse_2 = b.cos();   // cos(b) — used in entries [0,0] and [1,0]
    //  └─ Each trig function called exactly once

    [
        __cse_0 * __cse_2,   // [0,0]: cos(a)*cos(b)
        -__cse_1,            // [0,1]: -sin(a)
        __cse_1 * __cse_2,   // [1,0]: sin(a)*cos(b)
        __cse_0,             // [1,1]: cos(a)
    ]
}
```

Line by line:

1. **`#[must_use]`** — Warns if the caller discards the return value. Controlled by `CodegenOptions::must_use`.

2. **`pub fn rot_ab(a: f64, b: f64) -> [f64; 4]`** — Public function, named parameters matching the symbolic variables, returns a flat array sized for a 2×2 matrix (4 elements).

3. **CSE temporaries** — `__cse_0`, `__cse_1`, `__cse_2` are the extracted common subexpressions. Each expensive trig call happens exactly once.

4. **Array literal** — The matrix entries in row-major order. Each entry references only the CSE temps and parameters — no redundant computation.

## What Gets Folded

The code generator handles these special cases:

| Expression | Generated Code |
|-----------|---------------|
| Integer `n` | `n.0` (e.g., `3.0`) |
| Rational `p/q` | Decimal (e.g., `0.5`, `0.25`) |
| `π` | `std::f64::consts::PI` |
| `e` | `std::f64::consts::E` |
| `sin(0)` | `0.0` (folded at generation) |
| `cos(0)` | `1.0` (folded at generation) |
| `exp(0)` | `1.0` (folded at generation) |
| `sin(π)` | `0.0` (folded at generation) |
| `x^2` | `x.powi(2)` |
| `x^(1/2)` | `x.sqrt()` |
| `x^(-1)` | `x.powi(-1)` |
| `x^3.5` | `x.powf(3.5)` |
| `atan2(y, x)` | `y.atan2(x)` |
| `min(a, b)` | `a.min(b)` |
| `max(a, b)` | `a.max(b)` |
| `-1 * x` | `-x` (negation, not multiplication) |
| `x + (-3)` | `x - 3.0` (subtraction, not negative addition) |

## Compiled Functions: The Runtime Alternative

If you don't need generated source code — just a fast callable function at runtime — use `.compile()`:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let f = expr!(x^2 + sin(x));
let compiled = f.compile(&["x"]).unwrap();

// compiled: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>
let result = compiled(&[1.5]);
println!("f(1.5) = {result}");

// Use in a loop
for i in 0..1000 {
    let val = compiled(&[i as f64 * 0.001]);
    // ... hot path, no tree walking
}
```

`.compile()` vs `.to_rust_fn()`:

| | `.compile()` | `.to_rust_fn()` |
|---|---|---|
| Output | Runtime closure | Source code string |
| Speed | Fast (lambdified) | Fastest (LLVM-optimized) |
| Dependencies | Needs symplex at runtime | No runtime dependency |
| Use case | Interactive, prototyping | Production, `build.rs`, embedded |

## Handling Errors

Code generation can fail if the expression contains nodes that don't map to `f64` arithmetic:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// Free symbols that aren't in the args list → error
let f = expr!(x + y);
let result = f.to_rust_fn("broken", &["x"]);
assert!(result.is_err()); // "y" is not in args

// Imaginary unit → error (no complex codegen)
let f = ctx.i_unit();
let result = f.to_rust_fn("complex", &[]);
assert!(result.is_err());
```

Always handle the `Result`:

```rust
let ctx = Context::new();
syms!(ctx; x);
let f = expr!(x^2);
match f.to_rust_fn("safe", &["x"]) {
    Ok(code) => println!("{code}"),
    Err(e) => eprintln!("codegen failed: {e}"),
}
```

## Limitations

### No Complex Number Codegen

Expressions involving the imaginary unit `I` cannot be converted to Rust code. The code generator targets real-valued `f64`/`f32` functions only.

Workaround: decompose complex expressions into real and imaginary parts before codegen:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// Instead of codegen on a complex expression,
// generate separate functions for real and imaginary parts
let real_part = x.cos(); // Re(e^(ix)) = cos(x)
let imag_part = x.sin(); // Im(e^(ix)) = sin(x)

let re_code = real_part.to_rust_fn("euler_re", &["x"]).unwrap();
let im_code = imag_part.to_rust_fn("euler_im", &["x"]).unwrap();
```

### No Special Functions

Not all symbolic functions have `f64` implementations. The code generator handles:

✅ Supported: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `exp`, `ln`, `log` (base-10), `sqrt`, `abs`, `sign`, `floor`, `ceil`, `min`, `max`, `powi`, `powf`

❌ Not supported: `gamma`, `erf`, `erfc`, `beta`, `bessel_j`, `bessel_y`, `dirac_delta`, `heaviside`, `lambertw`, `factorial`

If your expression contains unsupported functions, `to_rust_fn` will return an error.

### No Piecewise Codegen (Partial)

Piecewise expressions generate `if`/`else` chains, but complex condition trees may produce verbose output. For control-flow-heavy expressions, consider hand-writing the numerical code and using symplex only for the mathematical derivation.

### No Matrix Struct Output

Matrix codegen returns flat arrays (`[f64; N]`), not structured matrix types. If you need to interface with `nalgebra` or `ndarray`, write a thin wrapper:

```rust
// Generated by symplex
pub fn rotation(theta: f64) -> [f64; 4] { /* ... */ }

// Your wrapper
fn rotation_nalgebra(theta: f64) -> nalgebra::Matrix2<f64> {
    let flat = rotation(theta);
    nalgebra::Matrix2::from_row_slice(&flat)
}
```

## Best Practices

### 1. Simplify Before Codegen

Simpler expressions produce cleaner, faster code:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let f = expr!(sin(x)^2 + cos(x)^2 + x);

// Without simplification: generates sin²+cos²+x
let code_raw = f.to_rust_fn("raw", &["x"]).unwrap();

// With simplification: generates x + 1
let code_clean = f.simplify().to_rust_fn("clean", &["x"]).unwrap();

println!("Raw:\n{code_raw}");
println!("Clean:\n{code_clean}");
```

### 2. Use Exact Rationals for Constants

Floating-point literals in source expressions propagate through to the output. Use exact rationals to keep the derivation clean:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; theta);

// Good: exact rational
let l = ctx.rational(3, 10); // exactly 0.3
let f = &l * &theta.sin();

// The generated code will use clean 0.3, not 0.30000000000000004
let code = f.to_rust_fn("good", &["theta"]).unwrap();
println!("{code}");
```

### 3. Name Variables Meaningfully

The variable names in your symbolic expressions become the parameter names in the generated function:

```rust
use symplex::prelude::*;

let ctx = Context::new();
// These names will appear verbatim in the generated code
let joint1 = ctx.var("joint_angle_1");
let joint2 = ctx.var("joint_angle_2");

let f = &joint1.sin() + &joint2.cos();
let code = f.to_rust_fn("fk_x", &["joint_angle_1", "joint_angle_2"]).unwrap();
// pub fn fk_x(joint_angle_1: f64, joint_angle_2: f64) -> f64 { ... }
```

### 4. Verify Generated Code

Always verify the generated code against the symbolic expression at a few test points:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let f = expr!(x^3 - 2*x + 1);
let code = f.to_rust_fn("poly", &["x"]).unwrap();

// Verify at several points
for val in [0, 1, 2, -1, 10] {
    let symbolic = f.eval_f64_with(&[(&x, val)]).unwrap();
    println!("f({val}) = {symbolic:.6} (verify against generated code)");
}
```

### 5. Generate Once, Commit the Result

For production code, generate the Rust source once (or in `build.rs`), review it, and commit it to your repository. This way:
- The generated code is visible in code review
- There's no build-time dependency on symplex
- CI doesn't need to re-derive the math

## Summary

| Task | Method | Output |
|------|--------|--------|
| Generate scalar function | `expr.to_rust_fn(name, args)` | `pub fn name(args) -> f64` |
| Generate with options | `expr.to_rust_fn_with_options(name, args, opts)` | Custom precision/backend |
| Generate matrix function | `matrix.to_rust_fn(name, args)` | `pub fn name(args) -> [f64; N]` |
| Compile to closure | `expr.compile(args)` | `Box<dyn Fn(&[f64]) -> f64>` |
| Extract CSE bindings | `expr.cse()` | `(Vec<(Ex, Ex)>, Ex)` |
| Preset: no_std | `CodegenOptions::no_std()` | cfg-gated, inline, f64 |
| Preset: embedded f32 | `CodegenOptions::embedded_f32()` | cfg-gated, inline, f32 |

---

*[← Chapter 7: Matrices](07-matrices.md) | [Back to Table of Contents](index.md)*