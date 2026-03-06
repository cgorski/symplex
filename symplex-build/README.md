# symplex-build

Build-time code generation for symplex — derive symbolic equations and emit optimized `no_std` Rust.

> **⚠️ Experimental:** This crate's API may change before 1.0. It is functional but has limited test coverage.

## Usage

Add to your firmware crate's `Cargo.toml`:

```toml
[build-dependencies]
symplex-build = { version = "0.1", path = "../symplex-build" }
```

Then in `build.rs`:

```rust
use symplex::prelude::*;
use symplex_build::CodeGen;

fn main() {
    // Define symbolic joint variables
    vars!(theta1, theta2);
    let l1 = symplex::rational(3, 10);  // 0.3m
    let l2 = symplex::rational(1, 4);   // 0.25m

    // Compute forward kinematics and Jacobian symbolically
    let (x, y, _z) = symplex::robotics::fk_position(&[
        (&theta1, &symplex::int(0), &l1, &symplex::int(0)),
        (&theta2, &symplex::int(0), &l2, &symplex::int(0)),
    ]);
    let j = symplex::matrix::jacobian(&[&x, &y], &[&theta1, &theta2]);

    // Generate optimized numerical Rust code
    CodeGen::new()
        .no_std(true)
        .add_matrix_fn("jacobian", &j, &["theta1", "theta2"])
        .write_to_out_dir("robot_math.rs")
        .unwrap();
}
```

Or use the convenience builder:

```rust
fn main() {
    symplex_build::robot_arm(&[
        ("theta1", 0.0, 0.3, 0.0),
        ("theta2", 0.0, 0.25, 0.0),
    ])
    .generate_all()
    .no_std()
    .write_to_out_dir("arm.rs")
    .unwrap();
}
```

Then in your library or binary crate:

```rust
include!(concat!(env!("OUT_DIR"), "/robot_math.rs"));
```

## Features

- **`no_std` support:** Generated code uses a cfg-gated math module that delegates to `libm` when `std` is unavailable.
- **Scalar and matrix functions:** Generate individual scalar functions or flat-array matrix functions.
- **Test generation:** Optionally emit `#[cfg(test)]` companion tests that verify generated functions produce finite results at configured test points.
- **TOML config:** Load robot definitions from a TOML file with `from_toml()`.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.