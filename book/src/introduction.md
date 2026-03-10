# Introduction

symplex is a symbolic mathematics library for Rust. It represents mathematical expressions as exact symbolic objects — not floating-point approximations — and provides operations for differentiation, integration, equation solving, simplification, series expansion, and code generation.

## Who This Is For

symplex is designed for Rust developers who need symbolic computation as part of a larger system. Typical use cases include:

- **Robotics and control systems.** Derive a Jacobian or transfer function symbolically, then generate optimized Rust code that runs in a real-time control loop. The library includes Denavit-Hartenberg parameter support, Lagrangian dynamics, state-space models, and Laplace transforms.

- **Code generation from mathematics.** Write a formula once as a symbolic expression, differentiate it, simplify, and emit an optimized Rust function with common subexpression elimination. The generated code uses `mul_add` and `powi` — it is production-grade, not a toy.

- **Physics and engineering with units.** symplex provides compile-time dimensional analysis: adding a `Mass` to a `Length` is a compiler error, and differentiating `Length` with respect to `Time` produces `Velocity`. This catches entire classes of bugs before the code runs.

- **Numerical methods development.** Derive finite-difference stencils, verify integration formulas, or compute Taylor series symbolically, then evaluate numerically with arbitrary precision.

## What It Provides

The core operations:

| Category | Capabilities |
|----------|-------------|
| Calculus | Differentiation (all elementary functions), integration (15+ strategies including Risch), limits (Gruntz algorithm), Taylor/Laurent/formal power series |
| Algebra | Expansion, factoring over ℤ, GCD, simplification (24 rules + Fu's trig algorithm), partial fractions, Gröbner bases |
| Solving | Polynomial through quartic by radicals, RootOf for degree ≥ 5, transcendental via Lambert W, systems via Gröbner bases, inequalities, 13 ODE classes |
| Linear algebra | Determinant, inverse, eigenvalues, Jordan form, matrix exponential, characteristic polynomial, Cholesky, LU |
| Transforms | Laplace (forward and inverse), Z-transform, Fourier, Gosper hypergeometric summation |
| Number theory | Primality testing, integer factorization, Euler's totient, Möbius function, CRT, modular arithmetic |
| Combinatorics | Stirling numbers (both kinds), Bell, Catalan, Fibonacci, multinomial coefficients, integer partition counting |
| Special functions | Gamma, Beta, erf, Bessel J/Y/I/K, Lambert W, Legendre, Chebyshev, Hermite, Laguerre |
| Output | Rust code generation with CSE, LaTeX rendering, JSON serialization, compiled closures for fast evaluation |
| Units | 30 physical quantity types with compile-time dimension checking, ~100 unit conversions (all exact rationals) |

## What It Does Not Provide

Being clear about limitations is important for evaluating whether this library fits your needs.

- **Geometry, statistics, and tensor algebra** are not implemented. If you need these, SymPy is the more complete choice today.
- **PDE solving** is not available. ODE solving covers 13 classes; partial differential equations are out of scope for now.
- **Group theory** is limited. There is no permutation group, symmetric group, or abstract algebra module.
- **Interactive notebooks** are not part of the library. symplex is a Rust library, not an application. A basic REPL is available as an example (`cargo run --example repl`), but it is not comparable to a Jupyter + SymPy environment.
- **Test coverage**, while substantial (~2,500 tests including cross-validation against SymPy), is far less than what SymPy has accumulated over 30 years of development.

## Design Principles

These choices are deliberate and pervasive:

1. **Exact arithmetic.** All numbers are `Ratio<BigInt>`. The expression `1/3` is the exact rational one-third, not `0.33333...`. Floating-point numbers appear only when explicitly requested via `eval_f64()`.

2. **Explicit state.** Every expression belongs to a `Context`. There is no hidden global state. This makes the library safe for multi-tenant servers, concurrent computation, and deterministic testing.

3. **Type safety.** Numeric expressions (`Ex`), boolean expressions (`BoolEx`), and set-valued expressions (`SetEx`) are distinct Rust types. Passing a boolean to `sin()` is a compile error.

4. **Thread safety.** `Context` is `Clone` (Arc-based) and `Ex` is `Send + Sync`. Multiple threads can work with the same context without data races.

5. **Honest failure.** Operations that cannot produce a closed-form result return unevaluated symbolic forms. `∫x^x dx` returns `Integral(x^x, x)` — a truthful representation of the problem — rather than an incorrect value or a panic. Numerical operations return `Result`.

## How to Read This Book

- **[Getting Started](./getting-started/installation.md)** covers installation, creating your first expressions, and the key concepts you need to be productive.
- **[Guide](./guide/calculus.md)** chapters are tutorial-style introductions to each domain (calculus, algebra, solving, matrices, code generation, units).
- **[Cookbook](./cookbook/pid-controller.md)** entries are worked solutions to real engineering and science problems.
- **[Reference](./reference/api-patterns.md)** documents the API conventions, error handling, and a migration guide for SymPy users.

For API documentation of individual functions and types, see [docs.rs/symplex](https://docs.rs/symplex).