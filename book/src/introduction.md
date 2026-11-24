# Introduction

symplex is a symbolic mathematics library for Rust. It represents mathematical expressions as exact symbolic objects — not floating-point approximations — and provides operations for differentiation, integration, summation, equation solving, simplification, series expansion, integral transforms, exact linear algebra and linear programming, proofs of polynomial inequalities with Lean export, probability and statistics, and code generation.

## Who This Is For

symplex is designed for Rust developers who need symbolic computation as part of a larger system. Typical use cases include:

- **Robotics and control systems.** Derive a Jacobian or transfer function symbolically, then generate optimized Rust or C code that runs in a real-time control loop. The library includes Denavit–Hartenberg parameter support, Lagrangian dynamics, state-space models, and Laplace/Fourier/Z transforms.

- **Code generation from mathematics.** Write a formula once as a symbolic expression, differentiate it, simplify, and emit an optimized Rust or C99 function with common subexpression elimination, `fma`, and a self-contained special-function runtime.

- **Physics and engineering with units.** symplex provides compile-time dimensional analysis: adding a `Mass` to a `Length` is a compiler error, and differentiating `Length` with respect to `Time` produces `Velocity`.

- **Numerical methods development.** Derive finite-difference stencils, verify integration formulas, compute Taylor series and formal power series symbolically, then evaluate numerically with arbitrary precision or adaptive quadrature.

## What It Provides

| Category | Capabilities |
|----------|-------------|
| Calculus | Differentiation (all elementary and special functions), indefinite integration (rule-based strategies, Risch for `exp`/`ln` towers, LRT log-to-real, a heuristic ansatz), definite and improper integration with divergence detection, adaptive Gauss–Kronrod quadrature, one-sided limits (Gruntz), Taylor/Laurent/asymptotic/formal power series, residues |
| Summation | Faulhaber, Gosper, telescoping, binomial sums, p-series (`ζ(2m)` exact), power-series recognition, infinite products, convergence tests |
| Algebra | Expansion, Berlekamp–Zassenhaus factoring over ℤ (any degree), multivariate factoring, resultants/discriminants, GCD, partial fractions, rational normal form (`ratsimp`), Gröbner bases, a public rewrite-rule engine with AC matching and tracing |
| Polynomials | `Poly` view with exact rational or symbolic coefficients, monomial/coefficient access, exact evaluation, coefficient matrices for certificate searches, exact sign of a polynomial on an interval (Sturm) |
| Solving | Polynomials through quartic by radicals, `RootOf`/`RootSum` beyond, transcendental via Lambert W, general periodic solutions, linear systems (unique/parametric/inconsistent), polynomial systems with algebraic solutions, damped Newton, inequalities, 16 ODE classes + initial-value problems, linear recurrences |
| Linear programming | Exact simplex over ℚ with shadow prices and Farkas infeasibility certificates; `feasible_nonneg` for non-negative combinations |
| Certificates | Proofs that a polynomial is non-negative on a box, a half-line, a parametric polyhedron or all of ℝⁿ (Handelman, Pólya, sums of squares), re-verified in exact arithmetic and exportable as Lean 4 / Mathlib theorems; exact polytopes (vertices, volume, cuts) |
| Sets & logic | Interval algebra with a normal form, three-valued membership/subset queries, `reduce_inequalities`, NNF/CNF/DNF, DPLL satisfiability, truth tables |
| Linear algebra | Determinant, inverse, eigenvalues (exact `RootOf` for irreducible cubics/quartics), Jordan form, matrix exponential/power/square root, QR, Cholesky, LDLᵀ, LU, Gram–Schmidt, structure tests, norms, least squares; over ℤ: Hermite and Smith normal forms, integer kernels, lattice determinants |
| Numerical optimisation | Brent/bisection/Newton root finding, Nelder–Mead, Brent and golden-section scalar minimisation, deterministic differential evolution, floating-point and exact least-squares polynomial fits, trapezoidal rule |
| Transforms | Laplace (forward/inverse, initial/final value), Fourier (three conventions), Mellin (with fundamental strip), Z, Fourier series on arbitrary intervals |
| Complex analysis | `re`/`im`/`conjugate`/`arg` honest about unknown realness, `as_real_imag`, polar form, complex infinity |
| Number theory | Pollard–Brent rho + ECM factorization, BPSW primality, modular square roots, discrete logarithms, primitive roots, continued fractions, Pell and other Diophantine equations, CRT |
| Combinatorics | Stirling numbers (both kinds), Bell, Catalan, derangements, Fibonacci/Lucas, Bernoulli/Euler numbers, multinomial coefficients, integer partitions |
| Special functions | Gamma, log-gamma, incomplete gammas, digamma/polygamma, the erf family, Beta, Bessel J/Y/I/K, Airy, Lambert W, elliptic K/E/F/Π, `expint`, Si/Ci/Shi/Chi/Ei/li, Fresnel, polylog, Riemann zeta, the classical and associated orthogonal polynomials — all with arbitrary-precision evaluation inside an error bound |
| Probability and statistics | Random variables over 22 distribution families and finite tables with exact moments, probabilities, CDF, quantile and entropy; statistics on data: hypothesis tests, estimation, ANOVA, regression, agreement and reliability, survival analysis, Markov chains |
| Algebraic numbers | ℚ(α) field arithmetic with exact zero/sign testing, minimal polynomials; a `RootSum` with a polynomial body evaluates exactly (Vieta's formulas) |
| Output | Rust and C99 code generation with CSE and an embedded special-function runtime, Python/NumPy/Julia expressions, compiled closures, LaTeX, MathML, Lean 4 terms, JSON serialization, plots (text/SVG/TikZ) |
| Units | 30 physical quantity types with compile-time dimension checking, about 120 unit constructors and conversions (exact rationals) |

## What It Does Not Provide

Being clear about limitations is important for evaluating whether this library fits your needs. The list below is the short version; [Known Limitations](./reference/known-limitations.md) is the full one, including what is known to be imperfect.

- **Geometry and tensor algebra** are not implemented. If you need these, SymPy is the more complete choice today. (Statistics is: see the `stats` chapter.)
- **PDE solving** is not available. ODE solving covers 16 classes; partial differential equations are out of scope for now.
- **Group theory** is limited. There is no permutation group, symmetric group, or abstract algebra module.
- **Hypergeometric / Meijer-G machinery** is absent; definite integration relies on antiderivatives, symmetry, residue sums, and a table of classical improper integrals.
- **Systems of nonlinear equations** are solved symbolically only when they are polynomial with finitely many solutions; otherwise there is a Newton iteration from a starting point (`solve_numeric_system`).
- **Interactive notebooks** are not part of the library. symplex is a Rust library, not an application. A basic REPL is available as an example (`cargo run --example repl`), and `symplex-wasm` exposes a `Session` API for the browser.
- **Test coverage**, while substantial (about 15,130 tests including cross-validation against SymPy, plus nightly property fuzzing), is far less than what SymPy has accumulated over 30 years of development.
- **Platforms.** Linux x86-64 is the only platform the test suite runs on.
- **A stable API.** Before 1.0 a minor release may break the API or change a result; the CHANGELOG lists every such change first.

## Design Principles

These choices are deliberate and pervasive:

1. **Exact arithmetic.** All numbers are `Ratio<BigInt>`. The expression `1/3` is the exact rational one-third, not `0.33333...`. Floating-point numbers appear only where you ask for them: `eval_f64()`, `compile()`, `integrate_numeric()`, and the routines documented as `f64` (numerical optimisation, sampling, plots, `numdist`, logistic and Cox regression).

2. **Explicit state.** Every expression belongs to a `Context`. There is no hidden global state. This makes the library safe for multi-tenant servers, concurrent computation, and deterministic testing.

3. **Type safety.** Numeric expressions (`Ex`), boolean expressions (`BoolEx`), and set-valued expressions (`SetEx`) are distinct Rust types. Passing a boolean to `sin()` is a compile error.

4. **Thread safety.** `Context` is `Clone` (Arc-based) and `Ex` is `Send + Sync`. Multiple threads can work with the same context without data races (one lock per context, so they take turns in it; separate contexts run in parallel).

5. **A refusal rather than a guess.** Operations that cannot produce a closed-form result return unevaluated symbolic forms. `∫x^x dx` returns `Integral(x^x, x)` — the problem as posed — rather than an incorrect value or a panic. `∫₋₁¹ dx/x²` is `Err(Divergent)`, not `−2`. `re(z)` stays `re(z)` unless `z` is known to be real. Numerical operations return `Result`.

6. **One domain: ℂ, principal branch.** A symbol without assumptions may be complex, and every rewrite preserves the value there: `ln x + ln y` and `√(x²)` stay as written until assumptions make `ln(xy)` and `|x|` true. See [The Domain Model](./getting-started/key-concepts.md#the-domain-model).

## How to Read This Book

- **[What's New since 0.13](./whats-new-since-0.13.md)** summarises the releases 0.14 to 0.60, behaviour changes first; the older "What's New" pages ([0.2](./whats-new-0.2.md) … [0.13 / 0.14](./whats-new-0.13.md)) tour the features in the order they arrived.
- **[Getting Started](./getting-started/installation.md)** covers installation, creating your first expressions, and the key concepts you need to be productive.
- **[Guide](./guide/calculus.md)** chapters are tutorial-style introductions to each domain. Their Rust blocks are compiled and run with the test suite (`cargo test --doc`), so the calls shown exist and the values in the comments were produced by the code.
- **[Cookbook](./cookbook/pid-controller.md)** entries are worked solutions to real engineering and science problems, backed by runnable examples in `examples/`.
- **[Reference](./reference/api-patterns.md)** documents the API conventions, error handling, the [known limitations](./reference/known-limitations.md), the migration pages for 0.2, 0.4, 0.7 and 0.12 (their before/after fragments are the only blocks not compiled), and a migration guide for SymPy users.

For API documentation of individual functions and types, see [docs.rs/symplex](https://docs.rs/symplex).
