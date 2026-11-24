# Installation

## Adding symplex to Your Project

```sh
cargo add symplex
```

Or add it directly to your `Cargo.toml`:

```toml
[dependencies]
symplex = "0.59"
```

## Minimum Supported Rust Version

symplex requires **Rust 1.93+** (Edition 2024). Check your version with:

```sh
rustc --version
```

If you need to update:

```sh
rustup update stable
```

## Platform Support

symplex is pure Rust with no C dependencies and no `unsafe` code, so it is expected to build wherever Rust does. What is tested is narrower:

- **Linux x86_64** — the only platform on which CI builds the crates and runs the tests.
- **WebAssembly** (`wasm32-unknown-unknown`) — CI type-checks the `symplex-wasm` crate for this target; the test suite does not run there.
- **macOS, Windows, Linux aarch64** — not tested in CI.

The dependencies carry permissive licences only: MIT, Apache-2.0 (some with the LLVM exception), BSD-2-Clause and Unicode-3.0 — the list `deny.toml` allows and `cargo deny` checks in CI. There are no LGPL, GPL, or proprietary dependencies.

## Feature Flags

symplex has no optional feature flags, by design: every capability is always present (CONTRIBUTING.md, "Feature flags: one crate, no knobs"). Only `symplex-wasm` has one, for its panic hook.

## Companion Crates

| Crate | Purpose |
|-------|---------|
| `symplex-macros` | The `expr!`, `matrix!`, `eq!`, `dim!`, `rule!` procedural macros (a dependency of `symplex`; you do not add it yourself) |
| [`symplex-build`](https://github.com/cgorski/symplex/tree/main/symplex-build) | Build-time code generation: run the CAS in `build.rs` and emit `no_std` Rust for firmware |
| [`symplex-wasm`](https://github.com/cgorski/symplex/tree/main/symplex-wasm) | `wasm-bindgen` bindings with a persistent `Session` for browser notebooks and demos |

## Dependencies

The library depends on the following crates, all of which are widely used in the Rust ecosystem:

| Crate | Purpose |
|-------|---------|
| `num-bigint`, `num-rational`, `num-traits`, `num-integer` | Arbitrary-precision arithmetic |
| `num-complex` | `Complex64` for complex `f64` values |
| `smallvec` | Stack-allocated small vectors for expression nodes |
| `rustc-hash`, `bitflags` | Hash maps for the expression arena; flag sets for assumptions |
| `parking_lot` | Fast reader-writer locks for thread-safe contexts |
| `astro-float` | Arbitrary-precision floating-point for numerical evaluation |
| `serde`, `serde_json` | JSON serialization of expressions |
| `tracing` | Optional structured logging (enable with `RUST_LOG=symplex=debug`) |
| `typenum` | Compile-time type-level integers for dimensional analysis |
| `thiserror` | Error type derivation |

The proc-macro crates (`symplex-macros`) additionally depend on `syn`, `quote`, and `proc-macro2` for the `expr!`, `matrix!`, `dim!`, and related macros.

## Verifying Your Installation

Create a file and run it to confirm everything works:

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2).diff(&x);
    println!("d/dx(x²) = {f}");
    // Should print: d/dx(x²) = 2*x
}
```

```sh
cargo run
```

## Tracing / Debug Output

symplex uses the `tracing` crate for structured logging. To see what the library is doing internally (simplification steps, integration strategy selection, etc.):

```sh
RUST_LOG=symplex=debug cargo run --example quickstart
```

This is useful for understanding why a particular computation produces the result it does, or for diagnosing performance issues.

## Next Steps

Continue to [First Steps](./first-steps.md) to create your first symbolic expressions.