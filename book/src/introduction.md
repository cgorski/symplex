# Introduction

symplex is a symbolic mathematics library for Rust. It represents mathematical expressions as exact symbolic objects — not floating-point approximations — and provides operations for differentiation, integration, summation, equation solving, simplification, series expansion, integral transforms, and code generation.

## Who This Is For

symplex is designed for Rust developers who need symbolic computation as part of a larger system. Typical use cases include:

- **Robotics and control systems.** Derive a Jacobian or transfer function symbolically, then generate optimized Rust or C code that runs in a real-time control loop. The library includes Denavit–Hartenberg parameter support, Lagrangian dynamics, state-space models, and Laplace/Fourier/Z transforms.

- **Code generation from mathematics.** Write a formula once as a symbolic expression, differentiate it, simplify, and emit an optimized Rust or C99 function with common subexpression elimination, `fma`, and a self-contained special-function runtime.

- **Physics and engineering with units.** symplex provides compile-time dimensional analysis: adding a `Mass` to a `Length` is a compiler error, and differentiating `Length` with respect to `Time` produces `Velocity`.

- **Numerical methods development.** Derive finite-difference stencils, verify integration formulas, compute Taylor series and formal power series symbolically, then evaluate numerically with arbitrary precision or adaptive quadrature.

## What It Provides

| Category | Capabilities |
|----------|-------------|
| Calculus | Differentiation (all elementary and special functions), indefinite integration (15+ strategies incl. Risch + LRT log-to-real), definite and improper integration with divergence detection, adaptive Gauss–Kronrod quadrature, one-sided limits (Gruntz), Taylor/Laurent/asymptotic/formal power series, residues |
| Summation | Faulhaber, Gosper, telescoping, binomial sums, p-series (`ζ(2m)` exact), power-series recognition, infinite products, convergence tests |
| Algebra | Expansion, Berlekamp–Zassenhaus factoring over ℤ (any degree), multivariate factoring, resultants/discriminants, GCD, partial fractions, Gröbner bases, a public rewrite-rule engine with AC matching and tracing |
| Solving | Polynomials through quartic by radicals, `RootOf`/`RootSum` beyond, transcendental via Lambert W, general periodic solutions, linear systems (unique/parametric/inconsistent), polynomial systems with algebraic solutions, damped Newton, inequalities, 16 ODE classes + initial-value problems, linear recurrences |
| Sets & logic | Interval algebra with a normal form, three-valued membership/subset queries, `reduce_inequalities`, NNF/CNF/DNF, DPLL satisfiability, truth tables |
| Linear algebra | Determinant, inverse, eigenvalues (exact `RootOf` for irreducible cubics/quartics), Jordan form, matrix exponential/power/square root, QR, Cholesky, LDLᵀ, LU, Gram–Schmidt, structure tests, norms, least squares |
| Transforms | Laplace (forward/inverse, initial/final value), Fourier (three conventions), Mellin (with fundamental strip), Z, Fourier series on arbitrary intervals |
| Complex analysis | `re`/`im`/`conjugate`/`arg` honest about unknown realness, `as_real_imag`, polar form, complex infinity |
| Number theory | Pollard–Brent rho + ECM factorization, BPSW primality, modular square roots, discrete logarithms, primitive roots, continued fractions, Pell and other Diophantine equations, CRT |
| Combinatorics | Stirling numbers (both kinds), Bell, Catalan, derangements, Fibonacci/Lucas, Bernoulli/Euler numbers, multinomial coefficients, integer partitions |
| Special functions | Gamma, log-gamma, digamma/polygamma, erf/erfc, Beta, Bessel J/Y/I/K, Lambert W, Si/Ci/Ei/li, Riemann zeta, Legendre/Chebyshev/Hermite/Laguerre polynomials — all with arbitrary-precision evaluation |
| Algebraic numbers | ℚ(α) field arithmetic with exact zero/sign testing, minimal polynomials, Vieta's formulas |
| Output | Rust and C99 code generation with CSE, compiled closures, LaTeX, JSON serialization, plots (text/SVG/TikZ) |
| Units | 30 physical quantity types with compile-time dimension checking, ~100 unit conversions (all exact rationals) |

## What It Does Not Provide

Being clear about limitations is important for evaluating whether this library fits your needs.

- **Geometry, statistics, and tensor algebra** are not implemented. If you need these, SymPy is the more complete choice today.
- **PDE solving** is not available. ODE solving covers 16 classes; partial differential equations are out of scope for now.
- **Group theory** is limited. There is no permutation group, symmetric group, or abstract algebra module.
- **Hypergeometric / Meijer-G machinery** is absent; definite integration relies on antiderivatives, symmetry, and a table of ~30 classical improper integrals.
- **Interactive notebooks** are not part of the library. symplex is a Rust library, not an application. A basic REPL is available as an example (`cargo run --example repl`), and `symplex-wasm` exposes a `Session` API for the browser.
- **Test coverage**, while substantial (~10,000 tests including cross-validation against SymPy), is far less than what SymPy has accumulated over 30 years of development.

## Design Principles

These choices are deliberate and pervasive:

1. **Exact arithmetic.** All numbers are `Ratio<BigInt>`. The expression `1/3` is the exact rational one-third, not `0.33333...`. Floating-point numbers appear only when explicitly requested via `eval_f64()`, `compile()`, or `integrate_numeric()`.

2. **Explicit state.** Every expression belongs to a `Context`. There is no hidden global state. This makes the library safe for multi-tenant servers, concurrent computation, and deterministic testing.

3. **Type safety.** Numeric expressions (`Ex`), boolean expressions (`BoolEx`), and set-valued expressions (`SetEx`) are distinct Rust types. Passing a boolean to `sin()` is a compile error.

4. **Thread safety.** `Context` is `Clone` (Arc-based) and `Ex` is `Send + Sync`. Multiple threads can work with the same context without data races.

5. **Honest failure.** Operations that cannot produce a closed-form result return unevaluated symbolic forms. `∫x^x dx` returns `Integral(x^x, x)` — a truthful representation of the problem — rather than an incorrect value or a panic. `∫₋₁¹ dx/x²` is `Err(Divergent)`, not `−2`. `re(z)` stays `re(z)` unless `z` is known to be real. Numerical operations return `Result`.

## How to Read This Book

- **[What's New in 0.2](./whats-new-0.2.md)** summarises the release for readers upgrading from 0.1.
- **[Getting Started](./getting-started/installation.md)** covers installation, creating your first expressions, and the key concepts you need to be productive.
- **[Guide](./guide/calculus.md)** chapters are tutorial-style introductions to each domain. Every code block is a complete program you can paste into `main.rs` (blocks marked `ignore` are fragments).
- **[Cookbook](./cookbook/pid-controller.md)** entries are worked solutions to real engineering and science problems, each backed by a runnable example in `examples/`.
- **[Reference](./reference/api-patterns.md)** documents the API conventions, error handling, the 0.1 → 0.2 migration, and a migration guide for SymPy users.

For API documentation of individual functions and types, see [docs.rs/symplex](https://docs.rs/symplex).
