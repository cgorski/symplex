# Symplex

A symbolic mathematics library for Rust — exact algebra, calculus, solving, simplification, and code generation with compile-time type safety and zero hidden global state.

[![Crates.io](https://img.shields.io/crates/v/symplex.svg)](https://crates.io/crates/symplex)
[![docs.rs](https://docs.rs/symplex/badge.svg)](https://docs.rs/symplex)
[![License](https://img.shields.io/crates/l/symplex.svg)](LICENSE-MIT)

> **Pre-release.** The API is unstable. Feedback welcome.
>
> Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture, conventions, and how to get started.

---

## Quick Start

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    syms!(ctx; x, y);

    // Differentiate
    let f = expr!(ctx, x^3 - 2*x + 1);
    println!("f'(x) = {}", f.diff(&x));          // 3*x^2 - 2

    // Integrate
    println!("∫f dx = {}", f.integrate(&x));      // 1/4*x^4 - x^2 + x

    // Solve
    let roots = expr!(ctx, x^2 - 5*x + 6).solve_or_empty(&x);
    println!("roots: {:?}", roots);                // [3, 2]

    // Simplify
    let trig = expr!(ctx, sin(x)^2 + cos(x)^2);
    println!("{}", trig.simplify());               // 1

    // Evaluate numerically
    let val = expr!(ctx, sin(x) + cos(x)).eval_f64_with(&[(&x, 1)]).unwrap();
    println!("{val:.6}");                          // 1.381773

    // Partial derivatives
    let g = expr!(ctx, x^2 * y + y^3);
    println!("∂g/∂x = {}", g.diff(&x));           // 2*x*y
    println!("∂g/∂y = {}", g.diff(&y));            // x^2 + 3*y^2

    // Matrices
    let m = matrix![ctx, [1, 2], [3, 4]];
    println!("det = {}", m.det().unwrap());        // -2

    // LaTeX
    println!("{}", f.to_latex());                  // x^{3} - 2 x + 1
}
```

```
cargo add symplex
```

---

## What You Can Do

### Calculus

```rust
let ctx = Context::new();
syms!(ctx; x);

// Differentiation — chain rule, product rule, all elementary functions
expr!(ctx, sin(x^2)).diff(&x);                    // 2*x*cos(x^2)

// Integration — by-parts, u-sub, partial fractions, trig, heurisch fallback
expr!(ctx, x * exp(x)).integrate(&x);             // x*exp(x) - exp(x)

// Limits — Gruntz algorithm for limits at infinity
expr!(ctx, sin(x) / x).limit(&x, &ctx.int(0));    // 1

// Taylor/Maclaurin series
expr!(ctx, exp(x)).series(&x, &ctx.int(0), 5);    // 1 + x + 1/2*x^2 + ...

// Formal power series with closed-form coefficient extraction
let fps = expr!(ctx, exp(x)).fps_maclaurin(&x);

// Laplace transforms (forward and inverse)
let s = ctx.symbol("s");
let t = ctx.symbol("t");
t.sin().laplace(&t, &s);                          // 1/(s^2 + 1)

// Fourier and Z-transforms
// Gosper hypergeometric summation
// Finite differences (Fornberg algorithm)
```

### Algebra

```rust
let ctx = Context::new();
syms!(ctx; x);

// Factoring
expr!(ctx, x^4 - 1).factor(&x);                   // (x - 1)*(x + 1)*(x^2 + 1)

// Expansion
expr!(ctx, (x + 1)^3).expand();                   // x^3 + 3*x^2 + 3*x + 1

// Simplification — 24 rules + Fu's trig + power/combinatorial strategies
expr!(ctx, (x^2 - 1) / (x - 1)).cancel(&x);       // x + 1

// Partial fractions
expr!(ctx, 1 / (x^2 - 1)).partial_fractions(&x);

// Polynomial GCD, LCM
// Gröbner bases (Buchberger + FGLM) for polynomial system solving
// Multivariate sparse polynomials
```

### Equation Solving

```rust
let ctx = Context::new();
syms!(ctx; x);

// Polynomial (linear through quartic via radicals, RootOf for degree ≥ 5)
expr!(ctx, x^2 - 5*x + 6).solve(&x);             // Ok([3, 2])

// Transcendental (Lambert W, inversion peeling, change of variable)
expr!(ctx, exp(x) - 5).solve(&x);                 // Ok([ln(5)])

// Systems via Gröbner bases
symplex::polysys::solve_system_ex(&[eq1, eq2], &[x, y]);

// Inequalities — sign-chart method
expr!(ctx, x^2 - 4).solve_gt(&x);                 // (-∞, -2) ∪ (2, ∞)

// ODE solving — 13 classes: separable, linear, Bernoulli, Euler-Cauchy,
// exact, integrating factor, variation of parameters, systems via matrix exp
```

### Linear Algebra

```rust
let ctx = Context::new();
let m = matrix![ctx, [1, 2], [3, 4]];

m.det().unwrap();                                  // -2
m.inv().unwrap();                                  // [[-2, 1], [3/2, -1/2]]
m.eigenvals(&ctx.symbol("λ")).unwrap();            // [-0.372..., 5.372...]
m.char_poly(&ctx.symbol("λ")).unwrap();            // λ^2 - 5*λ - 2

// Jordan normal form, diagonalization, matrix exponential
// Cholesky, LU decomposition, pseudo-inverse
// Kronecker product, rank, nullspace
```

### Transforms & Special Functions

```rust
let ctx = Context::new();
syms!(ctx; x, t, s);

// Laplace / inverse Laplace
t.exp().laplace(&t, &s);                          // 1/(s - 1)

// Bessel, Legendre, Chebyshev, Hermite, Laguerre
x.bessel_j(&ctx.int(0));                           // J_0(x)

// Gamma (arbitrary-precision Stirling), erf, Beta, LambertW
x.gamma();
x.erf();
x.lambertw();

// Digamma with arbitrary-precision evaluation
ctx.int(1).digamma().evalf(50);                    // -γ to 50 digits
```

### Code Generation

```rust
let ctx = Context::new();
syms!(ctx; x);
let f = expr!(ctx, x^3 - 2*x + 1);

// Generate optimized Rust function
let code = f.diff(&x).to_rust_fn("f_prime").unwrap();
// → pub fn f_prime(x: f64) -> f64 { 3_f64.mul_add(x.powi(2), -2_f64) }

// Common subexpression elimination
let (subs, result) = f.cse();

// LaTeX rendering
f.to_latex();                                      // x^{3} - 2 x + 1

// JSON serialization
f.to_json().unwrap();

// 2D Unicode pretty printing
f.pretty();
```

### Compile-Time Dimensional Analysis

```rust
use symplex::units::*;

let ctx = Context::new();
let m = Mass::symbol(&ctx, "m");
let a = Acceleration::symbol(&ctx, "a");
let h = Length::symbol(&ctx, "h");

// dim! macro: type-safe dimensional arithmetic
let force = dim!(ctx, Force: &m * &a);
let energy = dim!(ctx, Energy: &m * &a * &h);

// Mass + Length won't compile — dimension mismatch caught at compile time
// 30 named quantity types, ~100 unit conversions
// Typed calculus: d(Length)/d(Time) → Velocity
```

---

## Design Principles

1. **Exact by default.** Every number is `Ratio<BigInt>`. No floating-point contamination. `0.1 + 0.2 == 3/10`, not `0.30000000000000004`. Floats only appear on explicit `eval_f64()`.

2. **Explicit contexts.** Every expression belongs to a `Context`. No hidden global state. Mixing expressions from different contexts is caught immediately (compiler-enforced private field + runtime `checked_id` guard).

3. **Type-safe expressions.** `Ex` (numeric), `BoolEx` (boolean), `SetEx` (set-valued) are distinct types. `sin(bool_expr)` is a compile error.

4. **Thread-safe.** `Context` is `Clone` (Arc), `Ex` is `Send + Sync`. Multiple threads can share a context safely.

5. **No recursion.** All tree traversals use explicit stacks. Deep expressions don't blow the call stack.

6. **Never silently wrong.** Cross-context mixing panics with a clear message. Numerical evaluation returns `Result`. Unevaluated forms are honest — `∫x^x dx` returns `Integral(x^x, x)`, not garbage.

---

## The API Model

Every symbolic operation that might not produce a closed-form result has two entry points:

| Intent | Method | Returns | When to use |
|--------|--------|---------|-------------|
| Give me math | `integrate(&x)` | `Ex` (always — may contain `Integral` nodes) | Interactive exploration, chaining |
| Fail if you can't | `try_integrate(&x)` | `Result<Ex>` | Pipelines, codegen, safety-critical |

Check any expression for unevaluated forms:
```rust
let anti = hard_expr.integrate(&x);
if anti.has_unevaluated() {
    println!("integration produced formal result: {anti}");
}
```

Operations that always succeed (`simplify`, `expand`, `eval`, `factor`, `subs`) return `Ex` with no `try_` variant — "unchanged" is a valid answer.

Numeric boundary operations (`eval_f64`, `compile`, `to_rust_fn`) always return `Result` — crossing from symbols to numbers can fail if free symbols remain.

Queries (`is_positive`, `degree`, `equals`) return `Option<bool>` or `Option<T>` — three-valued: yes, no, or unknown.

---

## Comparison with SymPy

| Feature | symplex | SymPy |
|---------|---------|-------|
| Arithmetic | Exact `Ratio<BigInt>` | Exact (similar) |
| Differentiation | ✅ Complete | ✅ Complete |
| Integration | ✅ 15+ strategies | ✅ Risch + heurisch (broader) |
| Polynomial solving | ✅ Through quartic + RootOf | ✅ Through quartic + CRootOf |
| Series expansion | ✅ Taylor/Laurent/FPS | ✅ + O() notation |
| Limits | ✅ Gruntz algorithm | ✅ Gruntz (more mature) |
| Simplification | ✅ 24 rules + Fu | ✅ More strategies |
| Matrices | ✅ Eigenvalues, Jordan form | ✅ More decompositions |
| ODE solving | ✅ 13 classes | ✅ More classes |
| Laplace/Fourier | ✅ Table-based | ✅ Broader tables |
| Code generation | ✅ Optimized Rust with CSE | ✅ Python/C/Fortran |
| Dimensional analysis | ✅ Compile-time types | ❌ Not built-in |
| Thread safety | ✅ Send + Sync, no GIL | ❌ GIL-bound |
| Type safety | ✅ Ex/BoolEx/SetEx | ❌ Runtime only |
| Language | Rust (compiled) | Python (interpreted) |

**Where SymPy is stronger:** geometry, statistics, tensor algebra, quantum mechanics, combinatorics, Diophantine equations, PDE solving, and 30 years of community contributions.

**Where symplex is stronger:** compile-time type safety, thread safety, dimensional analysis with compile-time checking, optimized Rust code generation, and exact arithmetic without Python overhead.

---

## Examples

```
cargo run --example quickstart          # Core CAS operations
cargo run --example calculus            # Differentiation, integration, series, limits
cargo run --example equation_solving    # Polynomial, transcendental, system solving
cargo run --example matrix_algebra      # Eigenvalues, inverse, characteristic polynomial
cargo run --example ode_solving         # ODE classification and solving
cargo run --example laplace_transforms  # Forward/inverse Laplace transforms
cargo run --example complex_numbers     # Complex arithmetic and Euler's formula
cargo run --example number_theory       # Primality, factorization, CRT
cargo run --example optimization        # Gradient, Hessian, critical points
cargo run --example latex_output        # LaTeX rendering
cargo run --example solve_system        # Gröbner-based polynomial systems
cargo run --example control_system      # State-space, transfer functions
cargo run --example dynamics            # Lagrangian mechanics
cargo run --example robotics_codegen    # DH → Jacobian → Rust code
cargo run --example units_physics       # Dimensional analysis
cargo run --example repl                # Interactive REPL
```

---

## Dependencies

All MIT or Apache-2.0. No C bindings. No LGPL.

Core: `num-bigint`, `num-rational`, `num-traits`, `num-integer`, `smallvec`, `rustc-hash`, `bitflags`, `parking_lot`, `thiserror`, `astro-float`, `serde`, `serde_json`, `tracing`, `typenum`.

Proc macros: `syn`, `quote`, `proc-macro2`.

## Requirements

Rust 1.93+ (Edition 2024).

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENSE-APACHE).