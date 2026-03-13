# Symplex

Symbolic mathematics for Rust.

[![Crates.io](https://img.shields.io/crates/v/symplex.svg)](https://crates.io/crates/symplex)
[![docs.rs](https://docs.rs/symplex/badge.svg)](https://docs.rs/symplex)
[![License](https://img.shields.io/crates/l/symplex.svg)](LICENSE-MIT)

> **Pre-release.** The API is unstable. Feedback welcome.
>
> Contributing? See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture, conventions, and how to get started.

---

## What This Is

symplex is a symbolic computation library. It manipulates mathematical expressions exactly — using arbitrary-precision rational arithmetic, not floating-point — and can differentiate, integrate, solve equations, simplify, and generate optimized Rust code from symbolic results.

It is designed for Rust developers working in robotics, control systems, physics simulation, signal processing, or anywhere that symbolic math feeds into numerical code.

## Quick Example

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Build an expression and differentiate
    let f = expr!(ctx, x^3 - 2*x + 1);
    let df = f.diff(&x);
    println!("f'(x) = {df}");                      // 3*x^2 - 2

    // Solve an equation
    let roots = expr!(ctx, x^2 - 5*x + 6).solve_or_empty(&x);
    println!("roots: {roots:?}");                   // [3, 2]

    // Simplify a trig identity
    let trig = expr!(ctx, sin(x)^2 + cos(x)^2);
    println!("{}", trig.simplify());                // 1

    // Generate optimized Rust code from a symbolic result
    let code = df.to_rust_fn("gradient", &["x"]).unwrap();
    println!("{code}");
    // → pub fn gradient(x: f64) -> f64 { 3_f64.mul_add(x.powi(2), -2_f64) }

    // Or compile to a callable closure — no codegen, just fast evaluation
    let grad = df.compile(&["x"]).unwrap();
    println!("f'(2) = {}", grad(&[2.0]));           // 10.0
}
```

```
cargo add symplex
```

---

## When to Use This

- You need symbolic differentiation, integration, or equation solving and want to stay in Rust.
- You are generating numerical code from symbolic derivations — Jacobians, transfer functions, filter coefficients, control laws.
- You need compile-time dimensional analysis for physical quantities.
- You need thread-safe symbolic computation without a GIL or global interpreter lock.
- You want exact rational arithmetic (`1/3` stays as `1/3`, not `0.33333...`).

## When Not to Use This

- You need a mature CAS with decades of community validation — use [SymPy](https://www.sympy.org/). It has broader coverage, more special functions, and a much larger test corpus.
- You need geometry, statistics, tensor algebra, or PDE solving — these are not yet available.
- You need interactive notebook-style exploration — symplex is a library, not an application. (Though see `cargo run --example repl` for a basic REPL.)
- You need results verified against extensive known-answer databases — symplex has ~6,000 tests, but SymPy has orders of magnitude more coverage.

---

## What You Can Do

### Calculus

Differentiation handles the chain rule, product rule, and all elementary functions. Integration uses 15+ strategies including by-parts, u-substitution, partial fractions, trig substitution, Risch algorithm, Lazard-Rioboo-Trager log-to-real conversion, and heuristic integration. Radical coefficients (e.g., `√5` from cyclotomic denominators) are handled exactly via algebraic number field arithmetic.

```rust
let ctx = Context::new();
syms!(ctx; x);

// Differentiation
expr!(ctx, sin(x^2)).diff(&x);                    // 2*x*cos(x^2)

// Integration
expr!(ctx, x * exp(x)).integrate(&x);             // x*exp(x) - exp(x)

// Limits (Gruntz algorithm)
expr!(ctx, sin(x) / x).limit(&x, &ctx.int(0));    // 1

// Taylor / Maclaurin series
expr!(ctx, exp(x)).series(&x, &ctx.int(0), 5);    // 1 + x + x^2/2 + x^3/6 + x^4/24

// Laplace transforms (forward and inverse)
let (s, t) = (ctx.symbol("s"), ctx.symbol("t"));
t.sin().laplace(&t, &s);                          // 1/(s^2 + 1)

// Gosper hypergeometric summation
// Formal power series with closed-form coefficient extraction
// Finite differences (Fornberg algorithm)
```

### Algebra

Polynomial operations work over ℚ using arbitrary-precision rational arithmetic. Gröbner basis computation uses Buchberger's algorithm with FGLM order conversion.

```rust
let ctx = Context::new();
syms!(ctx; x);

// Factoring over ℤ
expr!(ctx, x^4 - 1).factor(&x);                   // (x - 1)*(x + 1)*(x^2 + 1)

// Expansion
expr!(ctx, (x + 1)^3).expand();                   // x^3 + 3*x^2 + 3*x + 1

// Common-factor cancellation
expr!(ctx, (x^2 - 1) / (x - 1)).cancel(&x);       // x + 1

// Partial fraction decomposition
// Polynomial GCD, resultant, square-free factorization
// Gröbner bases for polynomial system solving
// Multivariate sparse polynomials with pluggable monomial orderings
```

### Equation Solving

Polynomial equations are solved through quartic by radicals. Degree ≥ 5 produces `RootOf` nodes with numerical evaluation. Transcendental equations use inversion peeling and Lambert W.

```rust
let ctx = Context::new();
syms!(ctx; x);

// Polynomial solving
expr!(ctx, x^2 - 5*x + 6).solve(&x);             // Ok([3, 2])

// Transcendental
expr!(ctx, exp(x) - 5).solve(&x);                 // Ok([ln(5)])

// Systems via Gröbner bases
symplex::polysys::solve_system_ex(&[eq1, eq2], &[x, y]);

// Inequalities (sign-chart method)
expr!(ctx, x^2 - 4).solve_gt(&x);                 // (-∞, -2) ∪ (2, ∞)

// ODE solving — 13 classes: separable, linear, Bernoulli, Euler-Cauchy,
// exact, integrating factor, variation of parameters, systems via matrix exp
```

### Linear Algebra

Symbolic matrices support eigenvalue computation, Jordan normal form, matrix exponential, and characteristic polynomial extraction.

```rust
let ctx = Context::new();
let m = matrix![ctx, [1, 2], [3, 4]];

m.det().unwrap();                                  // -2
m.inv().unwrap();                                  // [[-2, 1], [3/2, -1/2]]
m.eigenvals(&ctx.symbol("λ")).unwrap();
m.char_poly(&ctx.symbol("λ")).unwrap();            // λ^2 - 5*λ - 2

// Jordan normal form, diagonalization, matrix exponential
// Cholesky, LU, pseudo-inverse
// Kronecker product, rank, nullspace
```

### Special Functions and Transforms

The library includes Bessel functions, Gamma, erf, Beta, Lambert W, and orthogonal polynomial families (Legendre, Chebyshev, Hermite, Laguerre). All support arbitrary-precision numerical evaluation.

```rust
let ctx = Context::new();
syms!(ctx; x, t, s);

// Laplace / inverse Laplace
t.exp().laplace(&t, &s);                          // 1/(s - 1)

// Bessel, Gamma, erf, Beta, LambertW
x.bessel_j(&ctx.int(0));
x.gamma();
x.erf();

// Arbitrary-precision evaluation
ctx.int(1).digamma().eval_decimal(50).unwrap();    // -γ to 50 digits
```

### Combinatorics and Number Theory

Stirling numbers, multinomial coefficients, integer partition counting, primality testing (deterministic Miller-Rabin), factorization, modular arithmetic, and the Chinese Remainder Theorem.

```rust
use symplex::combinatorics::*;
use symplex::ntheory::*;

// Stirling numbers of the second kind
stirling2(10, 4);                                  // Some(34105)

// Integer partitions
partition_count(100);                              // Some(190569292)

// Primality and factorization (works for i64 and BigInt)
isprime(104729);                                   // true
factorint(360);                                    // [(2,3), (3,2), (5,1)]

// Modular arithmetic
mod_inverse(17, 43);                               // Some(38)
crt_i64(&[2, 3, 2], &[3, 5, 7]);                  // Some(23)
```

### Code Generation

Symbolic expressions compile to optimized Rust functions with common subexpression elimination. The generated code uses `mul_add` and `powi` for numerical stability and performance.

```rust
let ctx = Context::new();
syms!(ctx; x);
let f = expr!(ctx, x^3 - 2*x + 1);

// Generate a Rust function as a String
let code = f.diff(&x).to_rust_fn("f_prime", &["x"]).unwrap();
// → pub fn f_prime(x: f64) -> f64 { 3_f64.mul_add(x.powi(2), -2_f64) }

// Common subexpression elimination
let (subs, result) = f.cse();

// Compile to a callable closure (no codegen, no file I/O)
let compiled = f.compile(&["x"]).unwrap();
assert!((compiled(&[3.0]) - 22.0).abs() < 1e-10);

// LaTeX rendering
f.to_latex();                                      // x^{3} - 2 x + 1
```

### Compile-Time Dimensional Analysis

Physical quantity types are checked at compile time. Adding a `Mass` to a `Length` is a compiler error. Differentiation respects dimensions: `d(Length)/d(Time)` produces `Velocity`.

```rust
use symplex::units::*;

let ctx = Context::new();
let m = Mass::symbol(&ctx, "m");
let a = Acceleration::symbol(&ctx, "a");

// dim! macro: type-safe dimensional arithmetic
let force = dim!(ctx, Force: &m * &a);             // F = m·a [N]

// Typed calculus: d(Length)/d(Time) → Velocity
let t = Time::symbol(&ctx, "t");
let position = Length::from_ex(expr!(ctx, 1/2 * a * t^2));
let velocity: Velocity = position.diff_wrt(&t);    // a·t [m/s]

// 30 named quantity types, ~100 unit conversions (all exact rationals)
// Mass + Length → compile error
```

---

## Design Principles

1. **Exact by default.** Every number is `Ratio<BigInt>`. No floating-point contamination. `0.1 + 0.2 == 3/10`, not `0.30000000000000004`. Floats only appear on explicit `eval_f64()`.

2. **Explicit contexts.** Every expression belongs to a `Context`. No hidden global state. Mixing expressions from different contexts is caught immediately (compiler-enforced private field + runtime guard).

3. **Type-safe expressions.** `Ex` (numeric), `BoolEx` (boolean), `SetEx` (set-valued) are distinct types. `sin(bool_expr)` is a compile error.

4. **Thread-safe.** `Context` is `Clone` (Arc-based), `Ex` is `Send + Sync`. Multiple threads can share a context safely.

5. **No recursion.** All tree traversals use explicit stacks. Deep expressions don't blow the call stack.

6. **Never silently wrong.** Numerical evaluation returns `Result`. Operations that can't produce a closed form return unevaluated symbolic nodes — `∫x^x dx` returns `Integral(x^x, x)`, not garbage.

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

Numeric boundary operations (`eval_f64`, `to_rust_fn`) always return `Result` — crossing from symbols to numbers can fail if free symbols remain. The `compile` method returns `Option` — `None` when the expression contains constructs that cannot be numerically evaluated.

Queries (`is_positive`, `degree`, `equals`) return `Option<bool>` or `Option<T>` — three-valued: yes, no, or unknown.

---

## Comparison with SymPy

| Feature | symplex | SymPy |
|---------|---------|-------|
| Arithmetic | Exact `Ratio<BigInt>` | Exact (similar) |
| Differentiation | Complete | Complete |
| Integration | 15+ strategies including Risch + LRT log-to-real | Risch + heurisch (broader) |
| Polynomial solving | Through quartic + RootOf | Through quartic + CRootOf |
| Series expansion | Taylor / Laurent / FPS | + O() notation |
| Limits | Gruntz algorithm | Gruntz (more mature) |
| Simplification | 24 rules + Fu's trig algorithm | More strategies |
| Matrices | Eigenvalues, Jordan form, exp | More decompositions |
| ODE solving | 13 classes | More classes |
| Laplace / Z-transforms | Table-based | Broader tables |
| Combinatorics | Stirling, Bell, partitions, multinomial | Broader (permutation groups, etc.) |
| Number theory | Primality, factorization, CRT, modular | Broader (Diophantine, quadratic forms) |
| Radical simplification | Construction-time (`√2·√3 → √6`) | Construction-time (similar) |
| Algebraic numbers | `ℚ(α)` field with exact zero/sign testing | `AlgebraicNumber` + `ANP` |
| Code generation | Optimized Rust with CSE | Python / C / Fortran |
| Dimensional analysis | Compile-time type checking | Not built-in |
| Thread safety | `Send + Sync`, no GIL | GIL-bound |
| Expression type safety | `Ex` / `BoolEx` / `SetEx` at compile time | Runtime only |
| Language | Rust (compiled, ~103K lines) | Python (interpreted) |

**Where SymPy is stronger:** geometry, statistics, tensor algebra, quantum mechanics, Diophantine equations, PDE solving, and 30 years of community contributions and testing.

**Where symplex is different:** compile-time dimensional analysis, thread safety, Rust code generation with CSE, exact arithmetic without Python overhead, algebraic number field arithmetic with exact zero/sign testing, and construction-time radical simplification. Operations that can't complete return honest unevaluated forms (including `RootSum` for degree ≥ 5 polynomial root sums) rather than hanging.

---

## Examples

**Getting started:**
```
cargo run --example quickstart              # Tour of core operations
cargo run --example repl                    # Interactive expression evaluation
```

**Engineering workflows:**
```
cargo run --example pid_controller          # PID design → stability → Rust codegen
cargo run --example robotics_codegen        # DH parameters → Jacobian → optimized Rust
cargo run --example control_system          # State-space, transfer functions, pole placement
cargo run --example signal_filter           # Bilinear transform → digital filter → codegen
cargo run --example dynamics                # Lagrangian mechanics, equations of motion
cargo run --example inverse_kinematics      # 2-DOF IK via Gröbner bases
```

**Mathematics and science:**
```
cargo run --example calculus                # Differentiation, integration, limits, series
cargo run --example equation_solving        # Polynomial, transcendental, system solving
cargo run --example matrix_algebra          # Eigenvalues, Jordan form, codegen
cargo run --example ode_solving             # ODE classification and solving
cargo run --example combinatorics_counting  # Stirling numbers, partitions, multinomials
cargo run --example optimization            # Gradient, Hessian, critical points
cargo run --example complex_numbers         # Euler's formula, complex roots
cargo run --example laplace_transforms      # Forward, inverse, z-transforms
```

**Applied problems:**
```
cargo run --example gradient_descent        # Symbolic gradient → compiled optimization loop
cargo run --example crypto_rsa              # RSA with number theory primitives
cargo run --example number_theory           # Primality, factorization, CRT
```

**Dimensional analysis:**
```
cargo run --example units_physics           # Compile-time unit checking
cargo run --example units_electrical        # Circuit analysis with units
cargo run --example units_engineering       # Motor design, imperial conversions
cargo run --example units_kinematics        # Kinematics with typed quantities
cargo run --example units_lagrangian        # Lagrangian mechanics with units
```

**Output:**
```
cargo run --example latex_output            # LaTeX rendering
cargo run --example physics_constants       # Physical constants (symbolic + exact)
```

---

## Dependencies

All MIT or Apache-2.0 licensed. No C bindings. No LGPL.

Core: `num-bigint`, `num-rational`, `num-traits`, `num-integer`, `smallvec`, `rustc-hash`, `bitflags`, `parking_lot`, `thiserror`, `astro-float`, `serde`, `serde_json`, `tracing`, `typenum`.

Proc macros: `syn`, `quote`, `proc-macro2`.

## Requirements

Rust 1.93+ (Edition 2024).

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENSE-APACHE).