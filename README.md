# Symplex

Symbolic mathematics library for Rust — calculus, algebra, Gröbner bases, robotics, control systems, and optimized code generation.

[![Crates.io](https://img.shields.io/crates/v/symplex.svg)](https://crates.io/crates/symplex)
[![docs.rs](https://docs.rs/symplex/badge.svg)](https://docs.rs/symplex)
[![License](https://img.shields.io/crates/l/symplex.svg)](LICENSE-MIT)

---

## Flagship Example

Define a 2-DOF planar robot arm with DH parameters, derive its symbolic Jacobian,
and generate an optimized Rust function — all in 15 lines:

```rust
use symplex::prelude::*;
use symplex::robotics::*;
use symplex::matrix::jacobian;

vars!(theta1, theta2);
let l1 = symplex::rational(3, 10);
let l2 = symplex::rational(1, 4);

let (px, py, _) = fk_position(&[
    (&theta1, &symplex::int(0), &l1, &symplex::int(0)),
    (&theta2, &symplex::int(0), &l2, &symplex::int(0)),
]);

let jac = jacobian(&[&px, &py], &[&theta1, &theta2]);
let code = jac.to_rust_fn("jacobian", &["theta1", "theta2"]).unwrap();
// → Optimized Rust function with common subexpression elimination
```

## Quick Start

```rust
use symplex::prelude::*;

fn main() {
    vars!(x, y);

    // Differentiation, factoring
    let f = expr!(x^2 + 2*x + 1);
    println!("{}", f.diff(&x));           // 2*x + 2
    println!("{}", f.factor(&x));         // (x + 1)^2

    // Equation solving
    let roots = expr!(x^2 - 5*x + 6).solve_or_empty(&x);
    // roots: [3, 2]

    // Trig simplification
    println!("{}", expr!(sin(x)^2 + cos(x)^2).simplify());  // 1

    // Integration
    let anti = expr!(x^2).integrate(&x);
    println!("{anti}");                   // 1/3*x^3

    // Partial derivatives
    let g = expr!(x^2 * y + y^3);
    println!("∂g/∂x = {}", g.diff(&x));  // 2*x*y
    println!("∂g/∂y = {}", g.diff(&y));  // x^2 + 3*y^2

    // Matrices
    let m = matrix![[1, 2], [3, 4]];
    println!("det = {}", m.det());        // -2

    // LaTeX and numerical evaluation
    println!("{}", f.to_latex());         // x^{2} + 2 x + 1
    let val = expr!(sin(x) + cos(x)).eval_f64_with(&[(&x, 1)]).unwrap();
    println!("{val:.6}");                 // 1.381773
}
```

### Compile-Time Dimensional Analysis

```rust
use symplex::prelude::*;
use symplex::units::*;

// Types enforce physical dimensions — Mass + Length won't compile
let m = Mass::symbol("m");
let g = Acceleration::symbol("g");
let h = Length::symbol("h");

// dim! macro: type-safe, order-independent dimensional arithmetic
let weight = dim!(Force: &m * &g);

// Multi-term expressions work too
let energy = dim!(Energy: &m * &g * &h);

// Typed calculus: d(Length)/d(Time) → Velocity
let v: Velocity = h.diff_wrt(&Time::symbol("t"));

// ~100 unit conversions
let distance = Length::meters(3.0) + Length::feet(6.5);

// Compile-time formula verification
const_assert_dim!(Force = Mass * Acceleration, "F = ma dimension check");
```

## Installation

```
cargo add symplex
```

## Why Symplex

**`dim!` macro for dimensional arithmetic.** All multiplication and division of
physical quantities goes through `dim!(OutputType: expr)` — a single proc macro that
is type-safe, order-independent, and gives clear error messages on dimension mismatches
via `FromDimExpr`.

**Exact by default.** Every number is a `Ratio<BigInt>` — `0.1 + 0.2` equals `3/10`,
not `0.30000000000000004`. Floating-point only appears when you explicitly ask for it
with `evalf()` or `lambdify()`.

**Type-safe expressions.** Numeric (`Ex`), boolean (`BoolEx`), and set (`SetEx`)
expressions are distinct types. Adding a boolean to a number is a compile error, not
a silent bug.

**Built for concurrency.** The arena uses `parking_lot::RwLock`; every `Ex` is
`Send + Sync`. Run symbolic pipelines across threads without a GIL or mutex
gymnastics.

**No recursion.** All tree traversals use explicit stacks. Deep expressions (thousands
of nested operations) won't blow the call stack.

**End-to-end robotics pipeline.** Go from DH parameters to simplified symbolic
Jacobians to optimized Rust source code in a single crate — no Python, no codegen
scripts, no glue.

## Proc Macros

Symplex provides five proc macros for natural math syntax:

```rust
vars!(x, y, z);                              // declare symbolic variables
let f = expr!(x^2 + 2*x + 1);               // build expressions
let g = expr!(sin(x)^2 + cos(x)^2);         // trig, exp, ln, sqrt, ...
let h = expr!(1/2 * x^2);                   // exact rationals
let b: BoolEx = expr!(x > 0 && y < 1);      // boolean expressions

let m = matrix![[x, 1], [0, x^2]];          // symbolic matrices
let eq = eq!(x^2 + x = 6);                  // equations (lhs = rhs)

// Dimensional arithmetic (units module)
let force = dim!(Force: &mass * &accel);     // type-safe multiply
let energy = dim!(Energy: &m * &g * &h);     // multi-term products
```

Constants `pi`, `E`, and `I` (imaginary unit) are available directly inside `expr!`.

## Features

**Core Algebra & Calculus:**

- ✅ Symbolic differentiation (chain rule, product rule, all elementary functions)
- ✅ Integration (by-parts, u-substitution, partial fractions, trig powers, cyclic IBP, heurisch fallback, parametric, Heaviside)
- ✅ Polynomial solving through quartic (Cardano/Ferrari) + Gröbner bases for systems
- ✅ Simplification (24 rules + Fu's trig simplification + power/combinatorial/numeric strategies)
- ✅ Taylor/Maclaurin/Laurent series, limits (Gruntz algorithm), formal power series
- ✅ Finite differences (Fornberg algorithm), Gosper hypergeometric summation
- ✅ Exact rational arithmetic (`Ratio<BigInt>`) — no floating-point contamination

**Linear Algebra:**

- ✅ Symbolic matrices: det, inverse, eigenvalues, LU/QR/Cholesky, matrix exponential
- ✅ Kronecker product, pseudo-inverse, rank, nullspace

**Robotics & Dynamics:**

- ✅ DH parameters, forward kinematics, Jacobian computation
- ✅ Lagrangian dynamics (Euler-Lagrange, mass/Coriolis/gravity matrices)
- ✅ Quaternion algebra for attitude representation
- ✅ 2-DOF algebraic inverse kinematics via Gröbner bases
- ✅ Optimized Rust code generation with cross-entry CSE

**Control Systems:**

- ✅ State-space and transfer function representations
- ✅ Routh-Hurwitz stability, Ackermann pole placement
- ✅ Zero-order hold discretization
- ✅ ODE solving: separable, linear, Bernoulli, Euler-Cauchy, variation of parameters, undetermined coefficients, systems via matrix exp

**Transforms & Special Functions:**

- ✅ Laplace/inverse Laplace, z-transform, Fourier transform
- ✅ Bessel, Legendre, Chebyshev, Hermite, Laguerre polynomials
- ✅ Arbitrary-precision Gamma (Stirling), erf, Beta

**Compile-Time Dimensional Analysis:**

- ✅ `dim!` proc macro for all dimensional multiplication/division: `dim!(Force: &m * &a)`
- ✅ 30 named physical quantity types (Force, Voltage, Energy, etc.)
- ✅ Compile-time dimension checking: `Mass + Length` won't compile
- ✅ `FromDimExpr` trait with `#[diagnostic::on_unimplemented]` for clear dimension mismatch errors
- ✅ Typed calculus: `d(Length)/d(Time) → Velocity` verified at compile time (DiffWrt/IntWrt)
- ✅ ~100 unit conversion constructors (meters, feet, horsepower, celsius, RPM)
- ✅ Runtime dimension inference for debug validation
- ✅ Typed robotics API: `fk_position_typed` with Angle/Length parameters
- ✅ Code generation with `uom` type annotations at function boundaries
- ✅ `const_assert_dim!` compile-time formula verification
- ✅ Physical constants (c, h, ℏ, k_B, N_A, G, g₀, e₀) with symbolic display and exact SI values

**Code Generation & Output:**

- ✅ `to_rust_fn()` with CodegenOptions (f32/f64, std/libm/no_std), FMA detection, Horner powi, sin_cos pairing, expm1/log1p/log2/exp2 optimization
- ✅ LaTeX rendering (`to_latex()` on expressions, matrices, quaternions)
- ✅ JSON serialization for interchange
- ✅ Runtime expression parser

**Architecture:**

- ✅ Arena hash-consing with O(1) structural equality
- ✅ Thread-safe (`Send + Sync`)
- ✅ No recursion (explicit stacks throughout)
- ✅ Type-safe: `Ex` (numeric) vs `BoolEx` (boolean) vs `SetEx` (sets)

## Examples

All 18 examples are self-contained and print annotated output.
Start with `quickstart` for a tour, or jump straight to `robotics_codegen`:

```
cargo run --example quickstart          # Basic CAS operations
cargo run --example equation_solving    # Polynomial + system solving
cargo run --example matrix_algebra      # Eigenvalues, inverse, char poly
cargo run --example calculus            # Differentiation, integration, series
cargo run --example optimization        # Gradient, Hessian, critical points
cargo run --example control_system      # State-space, transfer functions
cargo run --example dynamics            # Euler-Lagrange pendulum
cargo run --example robotics_codegen    # DH → Jacobian → Rust code
cargo run --example inverse_kinematics  # 2-DOF IK via Gröbner bases
cargo run --example latex_output        # LaTeX rendering
cargo run --example solve_system        # Gröbner-based polynomial systems
cargo run --example ode_solving         # ODE classification and solving
cargo run --example complex_numbers     # Complex arithmetic and Euler's formula
cargo run --example laplace_transforms  # Forward/inverse Laplace transforms
cargo run --example number_theory       # Primality, factorization, CRT
cargo run --example repl                # Interactive REPL
```

## Comparison with SymPy

A concise, honest comparison. For the full breakdown see
[`docs/COMPARISON.md`](docs/COMPARISON.md).

| Feature | symplex | SymPy |
|---------|:-------:|:-----:|
| Expression system | ✅ Arena hash-consing | ✅ Python objects |
| Differentiation | ✅ | ✅ |
| Integration | ✅ (elementary + heurisch) | ✅ (+ Risch) |
| Polynomial solving (through quartic) | ✅ | ✅ |
| Gröbner bases | ✅ Buchberger + FGLM | ✅ Buchberger + F5B |
| Polynomial system solving | ✅ | ✅ |
| Matrix algebra | ✅ 50+ methods | ✅ |
| Trig simplification (Fu's algorithm) | ✅ (17 transforms) | ✅ |
| ODE solving | ✅ (7 classes + systems) | ✅ |
| Hypergeometric summation (Gosper) | ✅ | ✅ |
| Formal power series | ✅ | ✅ |
| Finite differences (Fornberg) | ✅ | ✅ |
| Robotics (DH, FK, Jacobian, dynamics) | ✅ | ❌ (separate: mechanics) |
| Control systems | ✅ | ✅ (control module) |
| Rust code generation | ✅ | ❌ |
| Dimensional analysis | ✅ (compile-time, 30 types, `dim!` macro) | ❌ (separate: Pint) |
| LaTeX output | ✅ | ✅ |
| Thread safety | ✅ (`Send + Sync`) | ❌ (GIL) |
| Type-safe expressions | ✅ (`Ex` vs `BoolEx`) | ❌ |
| Number theory | Basic | ✅ Comprehensive |
| Statistics | ❌ | ✅ |
| Geometry | ❌ | ✅ |
| Tensor calculus | ❌ | ✅ |
| Python bindings | ❌ | N/A (native) |
| Plotting | ❌ | ✅ (matplotlib) |

**Bottom line:** symplex covers the core algebra → calculus → robotics → codegen
pipeline with Rust-native performance and safety guarantees. SymPy has 20+ years of
development and broader coverage in number theory, geometry, statistics, and tensor
calculus. Where both libraries overlap, symplex offers compile-time type safety,
zero-GIL parallelism, and native code generation that SymPy cannot.

## API Documentation

Full method-level documentation is on **[docs.rs/symplex](https://docs.rs/symplex)**.

Key entry points:

- [`symplex::prelude`](https://docs.rs/symplex/latest/symplex/prelude/) — `Ex`, `BoolEx`, `vars!`, `expr!`, `matrix!`, `eq!`
- [`symplex::units`](https://docs.rs/symplex/latest/symplex/units/) — compile-time dimensional analysis, 30 quantity types, typed calculus
- [`symplex::robotics`](https://docs.rs/symplex/latest/symplex/robotics/) — DH parameters, forward kinematics, inverse kinematics
- [`symplex::dynamics`](https://docs.rs/symplex/latest/symplex/dynamics/) — Euler-Lagrange, mass/Coriolis/gravity matrices
- [`symplex::control`](https://docs.rs/symplex/latest/symplex/control/) — state-space, transfer functions, stability
- [`symplex::quaternion`](https://docs.rs/symplex/latest/symplex/quaternion/) — quaternion algebra
- [`symplex::matrix`](https://docs.rs/symplex/latest/symplex/matrix/) — symbolic matrices, Jacobian, vector calculus

## Dependencies

All dependencies are MIT or Apache-2.0 licensed. No C bindings. No LGPL.

| Crate | Purpose |
|-------|---------|
| `num-bigint` / `num-rational` | Exact arbitrary-precision arithmetic |
| `parking_lot` | Fast reader-writer locks for the arena |
| `astro-float` | Arbitrary-precision floating-point evaluation |
| `serde` / `serde_json` | Serialization and JSON interchange |
| `symplex-macros` | Proc macros (`expr!`, `rule!`, `matrix!`, `eq!`, `dim!`) |

## Requirements

Rust 1.93+ (Edition 2024).

## Contributing

Contributions are welcome. Please open an issue before starting large changes.

```
cargo test                  # Run the full test suite (5,500+ tests)
cargo test --doc            # Doc-tests only
cargo bench                 # Benchmarks (criterion)
RUST_LOG=symplex=debug cargo run --example quickstart  # With tracing output
```

The crate uses zero-cost `tracing` instrumentation throughout. Enable structured
logging with the `RUST_LOG` environment variable to see simplification steps,
integration attempts, and Gröbner basis computation in real time.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.