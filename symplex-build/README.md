# symplex-build

Build-time code generation for [symplex](https://crates.io/crates/symplex) — run the CAS in `build.rs`, derive Jacobians, forward kinematics and other symbolic results, and emit optimized `no_std` Rust into `$OUT_DIR`.

> **⚠️ Experimental:** This crate's API may change before 1.0. It is functional but has limited test coverage.

## Usage

Add to your firmware crate's `Cargo.toml`:

```toml
[build-dependencies]
symplex-build = "0.3"
symplex = "0.3"
```

Then in `build.rs`:

```rust
use symplex::matrix::jacobian;
use symplex::prelude::*;
use symplex::robotics::{DhLink, fk_position};
use symplex_build::CodeGen;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; theta1, theta2);
    let zero = ctx.int(0);
    // Exact link lengths (0.3 m, 0.25 m) — no 0.30000000000000004.
    let l1 = ctx.rational(3, 10);
    let l2 = ctx.rational(1, 4);

    // Forward kinematics and Jacobian, symbolically.
    let (x, y, _z) = fk_position(&[
        DhLink { theta: &theta1, d: &zero, a: &l1, alpha: &zero },
        DhLink { theta: &theta2, d: &zero, a: &l2, alpha: &zero },
    ]);
    let j = jacobian(&[&x, &y], &[&theta1, &theta2]).unwrap();

    // Optimized numerical Rust with CSE, `mul_add`, and a cfg-gated math module.
    CodeGen::new()
        .no_std(true)
        .add_scalar_fn("fk_x", &x, &["theta1", "theta2"])
        .add_scalar_fn("fk_y", &y, &["theta1", "theta2"])
        .add_matrix_fn("jacobian", &j, &["theta1", "theta2"])
        .write_to_out_dir("robot_math.rs")
        .unwrap();
}
```

Or use the convenience builder — DH parameters given as `f64` are converted to the exact rational a human meant (`0.3` → `3/10`) via `Context::from_f64_approx`:

```rust
fn main() {
    symplex_build::robot_arm(&[
        ("theta1", 0.0, 0.3, 0.0),   // (theta_name, d, a, alpha)
        ("theta2", 0.0, 0.25, 0.0),
    ])
    .generate_all()                   // fk_x / fk_y / fk_z + jacobian
    .generate_fk_matrix("fk_t")       // full 4×4 homogeneous transform, [f64; 16] row-major
    .no_std()
    .write_to_out_dir("arm.rs")
    .unwrap();
}
```

Then in your library or binary crate:

```rust
include!(concat!(env!("OUT_DIR"), "/robot_math.rs"));
```

The generated file has no dependencies on `std` (with the `std` feature off it calls `libm`), so add to the firmware crate:

```toml
[features]
default = ["std"]
std = []

[dependencies]
libm = "0.2"          # used when `std` is disabled
```

## TOML configuration

`symplex_build::from_toml("robot.toml")` returns a pre-populated `CodeGen`:

```toml
[robot]
name = "two_link"

[[joints]]
theta = "theta1"
a = 0.3

[[joints]]
theta = "theta2"
a = 0.25

[generate]
functions = ["fk", "jacobian", "fk_matrix"]
output = "robot_math.rs"
```

## What is generated

- **Scalar functions**: `pub fn name(p1: f64, …) -> f64` with common subexpression elimination, `mul_add`, and `powi`.
- **Matrix functions**: `pub fn name(p1: f64, …) -> [f64; rows*cols]` (row-major) with CSE shared across all entries.
- **`no_std` math module**: with `.no_std(true)` (or `CodegenOptions::no_std()`), math calls go through a cfg-gated `mod math` emitted once at the top of the file. It covers every function the symplex backend (0.2 and 0.3) can emit — `sin`, `cos`, `tan`, `exp`, `ln`, `abs`, `sqrt`, `cbrt`, `asin`/`acos`/`atan`, `sinh`/`cosh`/`tanh`, `asinh`/`acosh`/`atanh`, `floor`, `ceil`, `signum`, `atan2`, `powf`, `powi`, `min`, `max`, `expm1`, `log1p`, `log2`, `exp2`, `fma`, `sin_cos` — with a `std` variant (inherent `f64`/`f32` methods) and a `libm` variant.
- **Companion tests**: `.with_tests(true).add_test_point(&[…])` appends a `#[cfg(test)] mod generated_tests` that evaluates every function at each point and asserts finite results.
- **Options**: `.options(CodegenOptions { … })` for `f32` precision (`.precision_f32()`), `#[inline]` (`.inline(true)`), `use_mul_add`, `checked_domain`, `uom` type annotations, etc.

## Special functions and the embedded runtime

Expressions that use `gamma`, `lgamma`, `digamma`, `erf`/`erfc`, Lambert W, Beta, Bessel functions, orthogonal polynomials or integer sequences compile to calls into a self-contained `mod symplex_rt { … }` that symplex embeds in the generated code. By default (`CodegenOptions::emit_runtime = true`) **every function embeds its own copy**, which is fine for one function but defines the module twice in a file with two such functions.

For multi-function files, emit the runtime once:

```rust
use symplex::matrix::{CodegenOptions, MathBackend};
use symplex::prelude::*;
use symplex_build::CodeGen;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let opts = CodegenOptions {
        math_backend: MathBackend::CfgGated,   // same as .no_std(true)
        emit_runtime: false,                   // functions reference symplex_rt:: but do not define it
        ..Default::default()
    };
    let body = CodeGen::new()
        .options(opts.clone())
        .add_scalar_fn("g", &x.gamma(), &["x"])
        .add_scalar_fn("w", &x.lambertw(), &["x"])
        .generate()
        .unwrap();
    // The complete runtime for this backend, containing every helper.
    let file = format!("{}\n{body}", opts.runtime_module());
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("special.rs");
    std::fs::write(out, file).unwrap();
    println!("cargo:rerun-if-changed=build.rs");
}
```

`runtime_module()` respects the math backend, so the `CfgGated` runtime is itself `no_std`-ready. (For C targets the same pattern applies with `Ex::to_c_fn_with_options` and `CodegenOptions::c_runtime()`.)

## Features

- **`no_std` support:** generated code uses a cfg-gated math module that delegates to `libm` when `std` is unavailable.
- **Exact parameters:** numeric DH parameters become reduced rationals, so generated constants are exact.
- **Scalar and matrix functions:** individual scalar functions or flat-array matrix functions (`fk_matrix` gives the 4×4 transform).
- **Test generation:** optional `#[cfg(test)]` companion tests at configured evaluation points.
- **TOML config:** load robot definitions from a TOML file with `from_toml()`.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
